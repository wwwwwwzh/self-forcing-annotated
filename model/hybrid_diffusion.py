from typing import Tuple
import torch

from model.diffusion import CausalDiffusion


class HybridCausalDiffusion(CausalDiffusion):
    def __init__(self, args, device):
        """
        Initialize the Hybrid Diffusion loss module.
        This uses a clean-prefix (teacher-forcing context) + noisy-suffix (diffusion forcing)
        within a single causal sequence.
        """
        super().__init__(args, device)
        self.hybrid_context_frames = getattr(args, "hybrid_context_frames", None)
        self.hybrid_context_blocks = getattr(args, "hybrid_context_blocks", None)

    def _flowmatch_noise_from_xt_x0(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        if timestep.ndim == 2:
            timestep_flat = timestep.flatten(0, 1)
        else:
            timestep_flat = timestep

        self.scheduler.sigmas = self.scheduler.sigmas.to(xt.device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(xt.device)
        timestep_id = torch.argmin(
            (self.scheduler.timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1)
        sigma = self.scheduler.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sigma = sigma.clamp_min(1e-8)

        xt_flat = xt.flatten(0, 1)
        x0_flat = x0.flatten(0, 1)
        noise_flat = (xt_flat - (1 - sigma) * x0_flat) / sigma
        return noise_flat.unflatten(0, xt.shape[:2])

    def _resolve_context_frames(self, num_frame: int) -> int:
        if self.hybrid_context_frames is not None:
            context_frames = int(self.hybrid_context_frames)
        elif self.hybrid_context_blocks is not None:
            context_frames = int(self.hybrid_context_blocks) * self.num_frame_per_block
        else:
            context_frames = max(num_frame - self.num_frame_per_block, self.num_frame_per_block)

        if context_frames <= 0 or context_frames >= num_frame:
            raise ValueError(
                f"Invalid hybrid context length: {context_frames} for total frames {num_frame}"
            )
        if context_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "hybrid_context_frames must be a multiple of num_frame_per_block"
            )
        return context_frames

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        ode_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Hybrid forcing:
          - clean prefix as causal context (teacher forcing)
          - noisy suffix with per-block timesteps (diffusion forcing)
        """
        if ode_latent is None:
            raise ValueError("ode_latent is required for hybrid training")

        if hasattr(self, "denoising_step_list"):
            expected_steps = len(self.denoising_step_list)
            if ode_latent.shape[1] != expected_steps:
                raise ValueError(
                    f"ode_latent has {ode_latent.shape[1]} steps, expected {expected_steps}"
                )

        # Override clean_latent to ensure it comes from ODE t=0
        clean_latent = ode_latent[:, -1].to(device=self.device, dtype=self.dtype)

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        context_frames = self._resolve_context_frames(num_frame)
        noisy_frames = num_frame - context_frames

        clean_context = clean_latent[:, :context_frames]
        clean_suffix = clean_latent[:, context_frames:]
        noise_context = noise[:, :context_frames]

        # Timesteps for the noisy suffix (diffusion forcing), draw from ODE trajectory
        if not hasattr(self, "denoising_step_list"):
            raise ValueError("denoising_step_list is required for hybrid training")

        index_suffix = self._get_timestep(
            0,
            len(self.denoising_step_list),
            image_or_video_shape[0],
            noisy_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        )

        # Gather noisy suffix from ODE trajectory
        ode_index = index_suffix.reshape(batch_size, 1, noisy_frames, 1, 1, 1).expand(
            -1, -1, -1, clean_latent.shape[2], clean_latent.shape[3], clean_latent.shape[4]
        ).to(self.device)
        noisy_suffix = torch.gather(
            ode_latent[:, :, context_frames:],
            dim=1,
            index=ode_index
        ).squeeze(1).to(dtype=self.dtype)

        timestep_suffix = self.denoising_step_list[index_suffix].to(
            device=self.device, dtype=self.scheduler.timesteps.dtype
        )

        # Compute flow-match target from xt/x0
        noise_suffix = self._flowmatch_noise_from_xt_x0(
            xt=noisy_suffix,
            x0=clean_suffix,
            timestep=timestep_suffix
        )
        training_target_suffix = self.scheduler.training_target(clean_suffix, noise_suffix, timestep_suffix)

        # Optional noise augmentation on the clean prefix
        if self.noise_augmentation_max_timestep > 0:
            index_ctx = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                context_frames,
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_context = self.scheduler.timesteps[index_ctx].to(dtype=self.dtype, device=self.device)
            clean_context_input = self.scheduler.add_noise(
                clean_context.flatten(0, 1),
                noise_context.flatten(0, 1),
                timestep_context.flatten(0, 1)
            ).unflatten(0, (batch_size, context_frames))
        else:
            timestep_context = torch.zeros(
                [batch_size, context_frames],
                device=self.device,
                dtype=self.scheduler.timesteps.dtype
            )
            clean_context_input = clean_context

        noisy_input = torch.cat([clean_context_input, noisy_suffix], dim=1)
        timestep = torch.cat([timestep_context, timestep_suffix], dim=1)

        # Train on full sequence, but compute loss only on the noisy suffix
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=None,
            aug_t=None
        )

        training_target = torch.zeros_like(flow_pred)
        training_target[:, context_frames:] = training_target_suffix

        per_frame_loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))

        weights = torch.zeros_like(timestep)
        weights[:, context_frames:] = self.scheduler.training_weight(timestep_suffix)
        weighted_loss = per_frame_loss * weights
        denom = weights[:, context_frames:].sum().clamp_min(1e-8)
        loss = weighted_loss.sum() / denom

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }
        return loss, log_dict

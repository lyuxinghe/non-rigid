# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from . import gaussian_diffusion_ddrd_joint as gd_ddrd_joint
from . import gaussian_diffusion_ddrd_seperate as gd_ddrd_separate
from . import gaussian_diffusion_v2 as gd_v2
from .respace import SpacedDiffusion, SpacedDiffusionDDRDJoint, SpacedDiffusionDDRDSeparate, SpacedDiffusionv2, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_ddrd_joint(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    time_based_weighting=False,
):
    betas = gd_ddrd_joint.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_ddrd_joint.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_ddrd_joint.LossType.RESCALED_MSE
    else:
        loss_type = gd_ddrd_joint.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusionDDRDJoint(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_ddrd_joint.ModelMeanType.EPSILON if not predict_xstart else gd_ddrd_joint.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_ddrd_joint.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_ddrd_joint.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_ddrd_joint.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        time_based_weighting=time_based_weighting,
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_ddrd_separate(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    time_based_weighting=False,
):
    betas = gd_ddrd_separate.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_ddrd_separate.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_ddrd_separate.LossType.RESCALED_MSE
    else:
        loss_type = gd_ddrd_separate.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusionDDRDSeparate(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_ddrd_separate.ModelMeanType.EPSILON if not predict_xstart else gd_ddrd_separate.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_ddrd_separate.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_ddrd_separate.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_ddrd_separate.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        time_based_weighting=time_based_weighting,
        # rescale_timesteps=rescale_timesteps,
    )

def create_diffusion_v2(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    time_based_weighting=False,
):
    betas = gd_v2.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd_v2.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_v2.LossType.RESCALED_MSE
    else:
        loss_type = gd_v2.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusionv2(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_v2.ModelMeanType.EPSILON if not predict_xstart else gd_v2.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_v2.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_v2.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_v2.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        time_based_weighting=time_based_weighting,
        # rescale_timesteps=rescale_timesteps,
    )
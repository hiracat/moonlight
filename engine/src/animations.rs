use proc_macros::{LuaUnion, LuaVal};
use ultraviolet::Slerp;

use crate::{
    components::{Time, Transform},
    ecs::{ReqM, World},
    resources::{Animated, Animation, AnimationResources, Keyframes, Mesh, SsboRegistry},
};

#[derive(LuaVal, Clone, Debug)]
#[lua(no_default)]
pub struct BlendTo {
    pub delay: f32,
    pub target: Animation,
}

#[derive(Debug, Clone, LuaUnion, Default)]
pub enum PlaybackMode {
    /// Play and loop immediately
    Loop(Animation),
    /// Play once, then blend back to bind pose
    Once(Animation),
    /// Play once and hold the last frame
    Clamp(Animation),
    /// Play current to completion, then switch to next with its own mode
    Then(Animation, Then),
    /// Play current to completion, then cross-fade into target over blend_duration
    Blend(Animation, Blend),
    /// No animation playing, joints lerp back to bind pose
    #[default]
    Stopped,
}

#[derive(Debug, Clone, LuaVal)]
#[lua(no_default)]
struct Then {
    next_mode: Box<PlaybackMode>,
}
#[derive(Debug, Clone, LuaVal)]
#[lua(no_default)]
struct Blend {
    blend_progress: f32,
    /// if > 0, wait this long before even starting the source
    delay: f32,
    next_mode: Box<PlaybackMode>,
}

pub fn update_animations(
    world: &mut World,
    animation_resources: &mut AnimationResources,
    ssbo_registry: &mut SsboRegistry,
) {
    let time = world.get_resource::<Time>().unwrap();
    let dt = time.delta_time;
    for entity in world.query_mut::<(ReqM<Mesh>, ReqM<Transform>, ReqM<Animated>)>() {
        let (_entityid, (mesh, _transform, animation)) = entity;
        if !mesh.animated {
            panic!("all entities with animated component's meshes should be animated meshes");
        }

        let skel_id = animation.skeleton.id;

        match &mut animation.mode {
            PlaybackMode::Stopped => {
                assert!(animation.blend_duration > 1e-10);
                animation.stop_blend_progress =
                    (animation.stop_blend_progress + dt).min(animation.blend_duration);
                let t = animation.stop_blend_progress / animation.blend_duration;

                for joint in &mut animation_resources.skeletons[skel_id].joints {
                    joint.position = joint.position * (1.0 - t) + joint.bind_position * t;
                    joint.rotation = joint.rotation.slerp(joint.bind_rotation, t);
                    joint.scale = joint.scale * (1.0 - t) + joint.bind_scale * t;
                }
            }

            PlaybackMode::Loop(current) => {
                let current_id = current.id;
                animation.time += dt * animation.speed;
                let duration = animation_duration(
                    &animation_resources.animations[skel_id][current_id].channels,
                );
                if animation.time >= duration {
                    animation.time -= duration;
                } else if animation.time < 0.0 {
                    animation.time += duration;
                }
                let channels = animation_resources.animations[skel_id][current_id]
                    .channels
                    .clone();
                apply_channels(
                    &channels,
                    &mut animation_resources.skeletons[skel_id].joints,
                    animation.time,
                );
            }

            PlaybackMode::Once(current) => {
                let current_id = current.id;
                animation.time += dt * animation.speed;
                let duration = animation_duration(
                    &animation_resources.animations[skel_id][current_id].channels,
                );
                if animation.time >= duration || animation.time < 0.0 {
                    animation.time = 0.0;
                    animation.stop_blend_progress = 0.0;
                    animation.mode = PlaybackMode::Stopped;
                    // fall through to next frame for bind pose blend
                } else {
                    let channels = animation_resources.animations[skel_id][current_id]
                        .channels
                        .clone();
                    apply_channels(
                        &channels,
                        &mut animation_resources.skeletons[skel_id].joints,
                        animation.time,
                    );
                }
            }

            PlaybackMode::Clamp(current) => {
                let current_id = current.id;
                animation.time += dt * animation.speed;
                let duration = animation_duration(
                    &animation_resources.animations[skel_id][current_id].channels,
                );
                animation.time = animation.time.clamp(0.0, duration);
                let channels = animation_resources.animations[skel_id][current_id]
                    .channels
                    .clone();
                apply_channels(
                    &channels,
                    &mut animation_resources.skeletons[skel_id].joints,
                    animation.time,
                );
            }

            PlaybackMode::Then(current, Then { next_mode }) => {
                let current_id = current.id;
                animation.time += dt * animation.speed;
                let duration = animation_duration(
                    &animation_resources.animations[skel_id][current_id].channels,
                );
                if animation.time >= duration || animation.time < 0.0 {
                    animation.time = 0.0;
                    animation.mode = *next_mode.clone();
                } else {
                    let channels = animation_resources.animations[skel_id][current_id]
                        .channels
                        .clone();
                    apply_channels(
                        &channels,
                        &mut animation_resources.skeletons[skel_id].joints,
                        animation.time,
                    );
                }
            }

            PlaybackMode::Blend(
                current,
                Blend {
                    blend_progress,
                    delay,
                    next_mode,
                },
            ) => {
                let current_id = current.id;

                if *delay > 0.0 {
                    // waiting before even starting — don't advance animation time yet
                    *delay -= dt;
                } else {
                    let duration = animation_duration(
                        &animation_resources.animations[skel_id][current_id].channels,
                    );
                    let source_finished = animation.time >= duration
                        || (animation.speed < 0.0 && animation.time < 0.0);

                    if !source_finished {
                        // source still playing normally
                        animation.time += dt * animation.speed;
                        let channels = animation_resources.animations[skel_id][current_id]
                            .channels
                            .clone();
                        apply_channels(
                            &channels,
                            &mut animation_resources.skeletons[skel_id].joints,
                            animation.time,
                        );
                    } else {
                        // source done — cross-fade into next_mode's animation
                        *blend_progress += dt;
                        let t = (*blend_progress / animation.blend_duration).min(1.0);

                        // extract the target animation from next_mode without consuming yet
                        let target_id = match next_mode.as_ref() {
                            PlaybackMode::Loop(a)
                            | PlaybackMode::Once(a)
                            | PlaybackMode::Clamp(a) => a.id,
                            PlaybackMode::Then(a, _) => a.id,
                            PlaybackMode::Blend(a, _) => a.id,
                            PlaybackMode::Stopped => {
                                // blending into stopped — just fade out
                                animation.stop_blend_progress = 0.0;
                                animation.mode = PlaybackMode::Stopped;
                                // continue outer entity loop
                                continue; // skip write, handled next frame
                            }
                        };

                        let src_channels = animation_resources.animations[skel_id][current_id]
                            .channels
                            .clone();
                        let tgt_channels = animation_resources.animations[skel_id][target_id]
                            .channels
                            .clone();

                        apply_channels(
                            &src_channels,
                            &mut animation_resources.skeletons[skel_id].joints,
                            duration, // hold last frame of source
                        );
                        blend_channels_toward(
                            &tgt_channels,
                            &mut animation_resources.skeletons[skel_id].joints,
                            *blend_progress, // target plays from its own t=0
                            t,
                        );

                        if t >= 1.0 {
                            animation.time = *blend_progress;
                            animation.mode = *next_mode.clone();
                        }
                    }
                }
            }
        }

        animation_resources.skeletons[skel_id].update_global_transforms();
        animation_resources.write_bones(ssbo_registry);
    }
}

/// Samples a single animation's channels at `time` and writes joint transforms directly.
fn apply_channels(
    channels: &[crate::resources::AnimationChannel],
    joints: &mut [crate::resources::Joint],
    time: f32,
) {
    for channel in channels {
        let next_idx = channel
            .timestamps
            .iter()
            .position(|&x| x > time)
            .unwrap_or(channel.timestamps.len() - 1);
        let cur_idx = next_idx.saturating_sub(1);

        let t1 = channel.timestamps[cur_idx];
        let t2 = channel.timestamps[next_idx];
        let time_between = t2 - t1;
        let percent = if time_between < 1e-6 {
            0.0
        } else {
            ((time - t1) / time_between).clamp(0.0, 1.0)
        };

        let joint = &mut joints[channel.target_joint_index];
        match &channel.keyframes {
            Keyframes::Translation(frames) => {
                let f1 = frames[cur_idx];
                let f2 = frames[next_idx];
                joint.position = f1 * (1.0 - percent) + f2 * percent;
            }
            Keyframes::Rotation(frames) => {
                joint.rotation = frames[cur_idx].slerp(frames[next_idx], percent);
            }
            Keyframes::Scale(frames) => {
                let f1 = frames[cur_idx];
                let f2 = frames[next_idx];
                joint.scale = f1 * (1.0 - percent) + f2 * percent;
            }
        }
    }
}
// Samples the target animation at `time` and blends each joint toward the result by `t` (0..=1).
fn blend_channels_toward(
    channels: &[crate::resources::AnimationChannel],
    joints: &mut [crate::resources::Joint],
    time: f32,
    strength: f32,
) {
    for channel in channels {
        let next_idx = channel
            .timestamps
            .iter()
            .position(|&x| x > time)
            .unwrap_or(channel.timestamps.len() - 1);
        let cur_idx = next_idx.saturating_sub(1);

        let t1 = channel.timestamps[cur_idx];
        let t2 = channel.timestamps[next_idx];
        let time_between = t2 - t1;
        let percent = if time_between < 1e-6 {
            0.0
        } else {
            ((time - t1) / time_between).clamp(0.0, 1.0)
        };

        let joint = &mut joints[channel.target_joint_index];
        match &channel.keyframes {
            Keyframes::Translation(frames) => {
                let target = frames[cur_idx] * (1.0 - percent) + frames[next_idx] * percent;
                joint.position = joint.position * (1.0 - strength) + target * strength;
            }
            Keyframes::Rotation(frames) => {
                let target = frames[cur_idx].slerp(frames[next_idx], percent);
                joint.rotation = joint.rotation.slerp(target, strength);
            }
            Keyframes::Scale(frames) => {
                let target = frames[cur_idx] * (1.0 - percent) + frames[next_idx] * percent;
                joint.scale = joint.scale * (1.0 - strength) + target * strength;
            }
        }
    }
}

// Returns the total duration of an animation (max of all channel end timestamps).
fn animation_duration(channels: &[crate::resources::AnimationChannel]) -> f32 {
    channels
        .iter()
        .flat_map(|c| c.timestamps.last().copied())
        .reduce(f32::max)
        .unwrap_or(0.0)
}

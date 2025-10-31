use ultraviolet::Vec3;

use crate::components::Transform;

#[derive(Debug, Copy, Clone)]
pub struct RigidBody {
    pub velocity: Vec3,
}
impl RigidBody {
    pub fn new() -> Self {
        Self {
            velocity: Vec3::zero(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Collider {
    Aabb(Aabb),
}
impl Collider {
    ///Panics: panics if any scale has negative components
    pub fn penetration_vector(
        from: &Collider,
        to: &Collider,
        from_tr: &Transform,
        to_tr: &Transform,
    ) -> Option<Vec3> {
        debug_assert!(from_tr.scale.x > 0.0 && from_tr.scale.y > 0.0 && from_tr.scale.z > 0.0);
        debug_assert!(to_tr.scale.x > 0.0 && to_tr.scale.y > 0.0 && to_tr.scale.z > 0.0);
        match (from, to) {
            (Collider::Aabb(from), Collider::Aabb(to)) => {
                let from_max = (from.max * from_tr.scale) + from_tr.position;
                let from_min = (from.min * from_tr.scale) + from_tr.position;
                let to_max = (to.max * to_tr.scale) + to_tr.position;
                let to_min = (to.min * to_tr.scale) + to_tr.position;

                let overlap_pos_x = from_max.x - to_min.x;
                let overlap_neg_x = to_max.x - from_min.x;

                let overlap_pos_y = from_max.y - to_min.y;
                let overlap_neg_y = to_max.y - from_min.y;

                let overlap_pos_z = from_max.z - to_min.z;
                let overlap_neg_z = to_max.z - from_min.z;

                if overlap_pos_x <= 0.0 || overlap_pos_y <= 0.0 || overlap_pos_z <= 0.0 {
                    return None;
                }
                if overlap_neg_x <= 0.0 || overlap_neg_y <= 0.0 || overlap_neg_z <= 0.0 {
                    return None;
                }

                let x;
                let y;
                let z;
                if overlap_pos_x >= overlap_neg_x {
                    x = overlap_neg_x
                } else {
                    x = -overlap_pos_x
                }
                if overlap_pos_y >= overlap_neg_y {
                    y = overlap_neg_y
                } else {
                    y = -overlap_pos_y
                }
                if overlap_pos_z >= overlap_neg_z {
                    z = overlap_neg_z
                } else {
                    z = -overlap_pos_z
                }
                if x.abs() < y.abs() && x.abs() < z.abs() {
                    Some(Vec3::new(x, 0.0, 0.0))
                } else if y.abs() < z.abs() {
                    Some(Vec3::new(0.0, y, 0.0))
                } else {
                    Some(Vec3::new(0.0, 0.0, z))
                }
            }
            _ => unimplemented!(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, target: Vec3) -> Self {
        Self {
            origin,
            dir: (target - origin).normalized(),
        }
    }
    pub fn from_direction(origin: Vec3, dir: Vec3) -> Self {
        Self {
            origin,
            dir: dir.normalized(),
        }
    }
    pub fn ray_box(ray: &Ray, collider: &Collider, collider_position: Vec3) -> Option<f32> {
        fn calculate_t_values(origin: f32, dir: f32, min: f32, max: f32) -> (f32, f32) {
            if dir.abs() < f32::EPSILON {
                if origin < min || origin > max {
                    return (f32::INFINITY, f32::NEG_INFINITY);
                } else {
                    return (f32::NEG_INFINITY, f32::INFINITY);
                }
            } else {
                let t1 = (min - origin) / dir;
                let t2 = (max - origin) / dir;
                return (t1.min(t2), t1.max(t2));
            }
        }
        match collider {
            Collider::Aabb(aabb) => {
                let aabb = aabb.at_position(collider_position);
                let (t_min_x, t_max_x) =
                    calculate_t_values(ray.origin.x, ray.dir.x, aabb.min.x, aabb.max.x);
                let (t_min_y, t_max_y) =
                    calculate_t_values(ray.origin.y, ray.dir.y, aabb.min.y, aabb.max.y);
                let (t_min_z, t_max_z) =
                    calculate_t_values(ray.origin.z, ray.dir.z, aabb.min.z, aabb.max.z);

                let t_enter = t_min_x.max(t_min_y).max(t_min_z);
                let t_exit = t_max_x.min(t_max_y).min(t_max_z);

                if t_exit < 0.0 || t_enter > t_exit {
                    return None;
                }

                let t_hit = if t_enter < 0.0 { t_exit } else { t_enter };
                Some(t_hit)
            }
            _ => unimplemented!(),
        }
    }
}

impl Aabb {
    pub fn new(half_extent: Vec3, offset: Vec3) -> Aabb {
        let local_max = offset + half_extent;
        let local_min = offset - half_extent;

        Aabb {
            min: local_min,
            max: local_max,
        }
    }

    pub fn at_position(&self, position: Vec3) -> Aabb {
        let global_max = position + self.max;
        let global_min = position + self.min;
        Aabb {
            min: global_min,
            max: global_max,
        }
    }
}
#[test]
fn ray_hits_aabb_center() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 0.0));
    let collider = Collider::Aabb(Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    });
    let collider_position = Vec3::zero();

    let result = Ray::ray_box(&ray, &collider, collider_position);

    assert!(result.is_some(), "Expected ray to hit the box");
    let t = result.unwrap();
    assert!(t > 0.0, "Expected hit to be in front of ray origin");
}

#[test]
fn ray_misses_aabb() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(5.0, 0.0, 0.0));
    let collider = Collider::Aabb(Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    });
    let collider_position = Vec3::zero();

    let result = Ray::ray_box(&ray, &collider, collider_position);

    assert!(result.is_none(), "Expected ray to miss the box");
}

#[test]
fn ray_starts_inside_aabb() {
    let ray = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let collider = Collider::Aabb(Aabb {
        min: Vec3::new(-1.0, -1.0, -1.0),
        max: Vec3::new(1.0, 1.0, 1.0),
    });
    let collider_position = Vec3::zero();

    let result = Ray::ray_box(&ray, &collider, collider_position);

    assert!(result.is_some(), "Expected ray to exit the box");
    let t = result.unwrap();
    assert!(t > 0.0, "Expected hit to be forward from the origin");
}

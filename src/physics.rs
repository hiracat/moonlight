use ultraviolet::Vec3;

#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}
impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }
}

#[derive(Clone, Copy)]
pub enum Collider {
    Aabb(Aabb),
}
impl Collider {
    pub fn penetration_vector(&self, other: Collider) -> Vec3 {
        match (self, other) {
            (Collider::Aabb(s), Collider::Aabb(o)) => {
                let dx1 = s.max.x - o.min.x;
                let dx2 = s.min.x - o.max.x;

                let dy1 = s.max.y - o.min.y;
                let dy2 = s.min.y - o.max.y;

                let dz1 = s.max.z - o.min.z;
                let dz2 = s.min.z - o.max.z;

                let dx = dx1.close_to_zero(dx2);
                let dy = dy1.close_to_zero(dy2);
                let dz = dz1.close_to_zero(dz2);

                Vec3 {
                    x: dx,
                    y: dy,
                    z: dz,
                }
            }
        }
    }
    pub fn intersects(&self, other: Collider) -> bool {
        match (self, other) {
            (Collider::Aabb(s), Collider::Aabb(o)) => {
                !(s.max.x < o.min.x
                    || s.min.x > o.max.x
                    || s.max.y < o.min.y
                    || s.min.y > o.max.y
                    || s.max.z < o.min.z
                    || s.min.z > o.max.z)
            }
        }
    }
}

trait CloseToZero {
    fn close_to_zero(self, other: Self) -> Self;
}

impl CloseToZero for f32 {
    fn close_to_zero(self, other: f32) -> f32 {
        if self.abs() < other.abs() {
            self
        } else {
            other
        }
    }
}

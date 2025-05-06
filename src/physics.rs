use ultraviolet::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}
impl Aabb {
    pub fn new(half_extent: Vec3, position: Vec3) -> Aabb {
        let global_max = position + half_extent;
        let global_min = position - half_extent;

        Aabb {
            min: global_min,
            max: global_max,
        }
    }
    pub fn translate(&mut self, delta: Vec3) {
        self.min += delta;
        self.max += delta;
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Collider {
    Aabb(Aabb),
}
impl Collider {
    pub fn penetration_vector(&self, other: &Collider) -> Vec3 {
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
    pub fn intersects(&self, other: &Collider) -> bool {
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
    pub fn translate(&mut self, delta: Vec3) {
        match self {
            Collider::Aabb(x) => {
                x.translate(delta);
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

use ultraviolet::Mat4;

use crate::resources;

pub struct Model {
    data: Vec<resources::Vertex>,
    translation: Mat4,
    rotation: Mat4,
    model: Mat4,
    normals: Mat4,
    requires_update: bool,
}

impl Model {
    pub fn model_matrices(&mut self) -> (Mat4, Mat4) {
        if self.requires_update {
            self.model = self.translation * self.rotation;
            self.normals = self.model.inversed().transposed();
            self.requires_update = false;
        }
        (self.model, self.normals)
    }
}

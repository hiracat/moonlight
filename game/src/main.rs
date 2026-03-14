use corruption::GameImpl;

use moonlight::core::App;
use ultraviolet::Vec3;

fn main() {
    let mut app = App::default();
    app.run(GameImpl::default());
}

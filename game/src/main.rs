use corruption::GameImpl;
use moonlight::core::App;

fn main() {
    let mut app = App::default();
    let game = GameImpl {};
    app.run(game);
}

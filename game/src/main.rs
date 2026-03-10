use moonlight::core::App;
use corruption::GameImpl;

fn main() {
    let mut app = App::default();
    let game = GameImpl {};
    app.run(game);
}

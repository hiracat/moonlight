use corruption::setup_game;

fn main() {
    let mut app = setup_game();
    app.run("data/scripts/main.lua");
}

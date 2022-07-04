
#include "stdafx.h"
#include "Game.h"


Game::Game() {
    // Run initialization
    this->initVariables();
}

void Game::initVariables() {
    // Initialize variables
    this->window = NULL;
    this->dt = 0.0f;
    srand(0);

    // Setup default options
    sf::VideoMode windowMode = sf::VideoMode::getDesktopMode();
    windowMode.width = 1400;
    windowMode.height = 1000;
    std::string title = "Genetic Algorithm";
    bool fullscreen = false;
    unsigned framerateLimit = 120;
    bool verticalSyncEnabled = false;

    // Setup window using default settings
    if (fullscreen) this->window = new sf::RenderWindow(windowMode, title, sf::Style::Fullscreen);
    else this->window = new sf::RenderWindow(windowMode, title, sf::Style::Titlebar | sf::Style::Close);
    this->window->setFramerateLimit(framerateLimit);
    this->window->setVerticalSyncEnabled(verticalSyncEnabled);

    // Initialize genepool
    this->genepool = VectorGenepoolSimulation({ 700.0f, 600.0f }, 4.0f, { 700.0f, 100.0f }, 20.0f, 4.0f, 500);
    this->genepool.initGenepool(1000, 0.01f);

    // Initialize UI
    this->uiManager = UIManager();
    float spacing = 6.0f;
    float size = 30.0f;
    this->uiManager.addElement(new UIButton(this->window, { spacing + 0 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "play.png",
        [this]() { this->genepool.startGeneration(); }
    ));
    this->uiManager.addElement(new UIButton(this->window, { spacing + 1 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "pause.png",
        [this]() { this->genepool.pauseGeneration(); }
    ));
    this->uiManager.addElement(new UIButton(this->window, { spacing + 2 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "complete.png",
        [this]() { this->genepool.fullStepGeneration(); }
    ));
    this->uiManager.addElement(new UIButton(this->window, { spacing + 3 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "finish.png",
        [this]() { this->genepool.finishGeneration(); }
    ));
    this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 4 * (spacing + size), spacing + 0 * (spacing + size) }, { size, size }, "hide.png", false,
        [this](bool toggled) { Globals::SHOW_VISUALS = !toggled; }
    ));
    this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 0 * (spacing + size), spacing + 1 * (spacing + size) }, { size, size }, "autoplay.png", false,
        [this](bool toggled) { this->genepool.setAutoStart(toggled); }
    ));
    this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 2 * (spacing + size), spacing + 1 * (spacing + size) }, { size, size }, "autocomplete.png", false,
        [this](bool toggled) { this->genepool.setAutoComplete(toggled); }
    ));
    this->uiManager.addElement(new UIToggleButton(this->window, { spacing + 3 * (spacing + size), spacing + 1 * (spacing + size) }, { size, size }, "autofinish.png", false,
        [this](bool toggled) { this->genepool.setAutoFinish(toggled); }
    ));
}

Game::~Game() {
    // Delete window
    delete this->window;
}


void Game::run() {
    // Run game loop
    while (this->window->isOpen()) {
        this->update();
        this->render();
    }
}

void Game::update() {
	// Update delta time using dtClock
	this->dt = this->dtClock.restart().asSeconds();

    // Poll Events
    while (this->window->pollEvent(this->sfEvent)) {
        switch (this->sfEvent.type) {

        // Closed window
        case sf::Event::Closed:
            this->window->close();
            break;
        }
    }

    // Update genepool
    this->genepool.update();
    this->uiManager.update();
}

void Game::render() {
    // Reset window background
    window->clear();

    // Render genepool
    if (Globals::SHOW_VISUALS) this->genepool.render(window);
    this->uiManager.render(window);

    // Display current draw window
    window->display();
}


#include "stdafx.h"
#include "global.h"


namespace global {

	bool showVisuals = true;
	sf::Font font;

	void initGlobal() {
		// Initialize global member variables
		font.loadFromFile("assets/arial.ttf");
	}
}

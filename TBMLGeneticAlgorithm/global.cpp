
#include "stdafx.h"
#include "global.h"

namespace global
{
	bool showVisuals = true;
	sf::Font font;

	void initialize()
	{
		font.loadFromFile("assets/arial.ttf");
	}
}

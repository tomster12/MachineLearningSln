#include "stdafx.h"
#include "global.h"

namespace global
{
	sf::Font font;

	void initialize()
	{
		font.loadFromFile("assets/arial.ttf");
	}
}

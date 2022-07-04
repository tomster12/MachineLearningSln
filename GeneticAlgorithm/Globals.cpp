
#include "stdafx.h"
#include "Globals.h"


// Initialize global static variables
bool Globals::SHOW_VISUALS = true;


class Globals GLOBALS;
void INIT_GLOBALS() {
	// Initialize global member variables
	GLOBALS.font.loadFromFile("arial.ttf");
}

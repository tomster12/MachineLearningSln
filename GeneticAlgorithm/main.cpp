
#include "stdafx.h"
#include "Game.h"


/* Setup

Project - General
Output Directory			$(SolutionDir)\bin\output\$(Configuration)\
Intermediate Directory		$(SolutionDir)\bin\intermediates\$(Configuration)\

Project - C / C++
Add.Include Directories		$(SolutionDir)\dependencies\SFML\include
Precompiled Header			Use (/Yu)

Project - Linker
Add.Library Directories		$(SolutionDir)\dependencies\SFML\lib; % (AdditionalLibraryDirectories)
Dbg. Add.Dependencies		sfml-system-d.lib;sfml-graphics-d.lib;sfml-audio-d.lib;sfml-network-d.lib;sfml-window-d.lib;%(AdditionalDependencies)
Rls. Add.Dependencies		sfml-system.lib;sfml-graphics.lib;sfml-audio.lib;sfml-network.lib;sfml-window.lib;%(AdditionalDependencies)

stdafx.cpp - C / C++
Precompiled Header			Create (/Yc)

*/


int main() {
	Game game;
	game.run();
	return 0;
}

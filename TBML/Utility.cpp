#include "Utility.h"

namespace tbml
{
	namespace fns
	{
		float getRandomFloat() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); }
	}
}

#pragma once

class DrawableGrid
{
public:
	DrawableGrid();
	DrawableGrid(sf::RenderWindow* window, int rows, int cols, int cellSize);
	~DrawableGrid();

	void update();
	void render();
	void setPosition(float x, float y);
	const std::vector<float>& getGrid() { return this->grid; }
	void subscribeToCellChange(std::function<void()> callback) { this->onGridChange.push_back(callback); }

private:
	sf::RenderWindow* window;
	int rows;
	int cols;
	int cellSize;
	float drawRadius;
	std::vector<float> grid;
	std::vector<sf::RectangleShape> cells;
	float x;
	float y;
	std::vector<std::function<void()>> onGridChange;
};

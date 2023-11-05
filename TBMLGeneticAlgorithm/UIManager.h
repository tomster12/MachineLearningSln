
#pragma once

class UIElement
{
public:
	virtual void update() {};
	virtual void render(sf::RenderWindow* window) {};
};

class UIButton : public UIElement
{
public:
	static const sf::Color HOVER_COLOR;

	UIButton(sf::RenderWindow* window, sf::Vector2f pos, sf::Vector2f size, std::string texturePath, std::function<void()> action);

	void update() override;
	void render(sf::RenderWindow* window) override;

private:
	sf::RenderWindow* window;
	sf::Vector2f pos;
	sf::Vector2f size;
	sf::Texture texture;
	sf::Sprite sprite;
	std::function<void()> action;

	bool isHovered;
	bool isPressed;
};

class UIToggleButton : public UIElement
{
public:
	static const sf::Color TOGGLE_COLOR;

	UIToggleButton(sf::RenderWindow* window, sf::Vector2f pos, sf::Vector2f size, std::string texturePath, bool initial, std::function<void(bool)> action);

	void update() override;
	void render(sf::RenderWindow* window) override;

private:
	sf::RenderWindow* window;
	sf::Vector2f pos;
	sf::Vector2f size;
	sf::Texture texture;
	sf::Sprite sprite;
	std::function<void(bool)> action;

	bool isToggled;
	bool isHovered;
	bool isPressed;
};

class UIDynamicText : public UIElement
{
public:
	UIDynamicText(sf::RenderWindow* window, sf::Vector2f pos, size_t fontSize, std::function<std::string(void)> textFunc);

	void update() override {};
	void render(sf::RenderWindow* window) override;

private:
	sf::RenderWindow* window;
	sf::Vector2f pos;
	sf::Text text;
	std::function<std::string(void)> textFunc;
};

class UIManager
{
public:
	void update();
	void render(sf::RenderWindow* window);

	void addElement(std::shared_ptr<UIElement>&& uiElement);

private:
	std::vector<std::shared_ptr<UIElement>> uiElements;
};

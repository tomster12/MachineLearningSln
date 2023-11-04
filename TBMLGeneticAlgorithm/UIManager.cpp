
#include "stdafx.h"
#include "UIManager.h"

#pragma region - UIButton

const sf::Color UIButton::HOVER_COLOR = sf::Color(100, 100, 100);

UIButton::UIButton(sf::RenderWindow* window, sf::Vector2f pos, sf::Vector2f size, std::string texturePath, std::function<void()> action)
	: window(window), pos(pos), size(size), action(action)
{

	// Initialize variables
	texture.loadFromFile(texturePath);
	sprite.setTexture(texture);
	sprite.setPosition(pos.x, pos.y);
	sprite.setScale(size.x / texture.getSize().x, size.y / texture.getSize().y);
	this->isHovered = false;
	this->isPressed = false;
}

void UIButton::update()
{
	// Check if mouse over
	sf::Vector2f mousePos = sf::Vector2f(sf::Mouse::getPosition(*this->window));
	this->isHovered = this->sprite.getGlobalBounds().contains(mousePos);

	// Call action when clicked
	if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
	{
		if (this->isHovered && !this->isPressed)
		{
			this->action();
			this->isPressed = true;
		}
	}
	else this->isPressed = false;

	// Highlight sprite is hovered
	if (this->isHovered) this->sprite.setColor(UIButton::HOVER_COLOR);
	else this->sprite.setColor(sf::Color::White);
}

void UIButton::render(sf::RenderWindow* window)
{
	// Update window to current
	this->window = window;

	// Draw sprite to window
	window->draw(sprite);
}

#pragma endregion

#pragma region - UIToggleButton

const sf::Color UIToggleButton::TOGGLE_COLOR = sf::Color(140, 140, 140);

UIToggleButton::UIToggleButton(sf::RenderWindow* window, sf::Vector2f pos, sf::Vector2f size, std::string texturePath, bool initial, std::function<void(bool)> action)
	: window(window), pos(pos), size(size), action(action)
{
	// Initialize variables
	texture.loadFromFile(texturePath);
	sprite.setTexture(texture);
	sprite.setPosition(pos.x, pos.y);
	sprite.setScale(size.x / texture.getSize().x, size.y / texture.getSize().y);
	this->isToggled = initial;
	this->isHovered = false;
	this->isPressed = false;
	action(this->isToggled);
}

void UIToggleButton::update()
{
	// Check if mouse over
	sf::Vector2f mousePos = sf::Vector2f(sf::Mouse::getPosition(*this->window));
	this->isHovered = this->sprite.getGlobalBounds().contains(mousePos);

	// Call action when clicked
	if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
	{
		if (this->isHovered && !this->isPressed)
		{
			this->isToggled = !this->isToggled;
			this->action(this->isToggled);
			this->isPressed = true;
		}
	}
	else this->isPressed = false;

	// Highlight sprite is hovered
	if (this->isHovered) this->sprite.setColor(UIButton::HOVER_COLOR);
	else if (this->isToggled) this->sprite.setColor(UIToggleButton::TOGGLE_COLOR);
	else this->sprite.setColor(sf::Color::White);
}

void UIToggleButton::render(sf::RenderWindow* window)
{
	// Update window to current
	this->window = window;

	// Draw sprite to window
	window->draw(sprite);
}

#pragma endregion

#pragma region - UIManager

UIManager::~UIManager()
{
	// Delete all UI elements
	for (auto& uiElement : uiElements) delete uiElement;
}

void UIManager::update()
{
	// Update all UI elements
	for (auto& uiElement : uiElements) uiElement->update();
}

void UIManager::render(sf::RenderWindow* window)
{
	// Render all UI elements
	for (auto& uiElement : uiElements) uiElement->render(window);
}

void UIManager::addElement(UIElement* uiElement)
{
	// Add to list of UI elements
	this->uiElements.push_back(uiElement);
}

#pragma endregion

-- AgroDashRD Database Schema

-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'AGRICULTOR',
    location VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Markets Table
CREATE TABLE markets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    location VARCHAR(255),
    type VARCHAR(50) DEFAULT 'Mayorista',
    is_active BOOLEAN DEFAULT TRUE
);

-- Products Table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100),
    unit VARCHAR(50) DEFAULT 'kg',
    image_url VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE
);

-- Prices Table
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    market_id INTEGER REFERENCES markets(id),
    reporter_id INTEGER REFERENCES users(id),
    date TIMESTAMP NOT NULL,
    price_wholesale FLOAT,
    price_retail FLOAT
);

-- Indexes for performance
CREATE INDEX idx_prices_date ON prices(date);
CREATE INDEX idx_prices_product ON prices(product_id);
CREATE INDEX idx_prices_market ON prices(market_id);

-- Initial Data (Optional)
INSERT INTO markets (name, type) VALUES
('Mercado Nuevo', 'Mayorista'),
('Merca Santo Domingo', 'Mayorista'),
('Supermercado Nacional', 'Supermercado');

INSERT INTO products (name, category, unit) VALUES
('Plátano Barahonero', 'Musáceas', 'Unidad'),
('Yuca Mocana', 'Raíces', 'Libra'),
('Tomate de Ensalada', 'Vegetales', 'Libra');

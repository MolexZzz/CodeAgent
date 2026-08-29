package com.example.vendingmachine.service;

import com.example.vendingmachine.model.Product;

import java.util.Map;

public class Inventory {
    private final Map<String, Product> products;
    private final Map<String, Integer> quantities;

    public Inventory(Map<String, Product> products, Map<String, Integer> quantities) {
        this.products = products;
        this.quantities = quantities;
    }

    public Map<String, Product> products() {
        return products;
    }

    public Map<String, Integer> quantities() {
        return quantities;
    }
}

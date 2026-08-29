package com.example.vendingmachine.model;

import java.util.Objects;

public record Product(String code, String name, int priceCents) {
    public Product {
        Objects.requireNonNull(code, "code");
        Objects.requireNonNull(name, "name");
    }
}

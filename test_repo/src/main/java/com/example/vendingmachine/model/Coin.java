package com.example.vendingmachine.model;

public enum Coin {
    NICKEL(5),
    DIME(10),
    QUARTER(25),
    ONE_DOLLAR(100);

    private final int valueCents;

    Coin(int valueCents) {
        this.valueCents = valueCents;
    }

    public int valueCents() {
        return valueCents;
    }
}

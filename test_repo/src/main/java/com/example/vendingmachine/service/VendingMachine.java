package com.example.vendingmachine.service;

import com.example.vendingmachine.model.Coin;
import com.example.vendingmachine.model.Product;
import com.example.vendingmachine.model.VendingMachineState;

import java.util.List;

public class VendingMachine {
    private final Inventory inventory;
    private int insertedCents;
    private Product selectedProduct;
    private VendingMachineState state = VendingMachineState.IDLE;

    public VendingMachine(Inventory inventory) {
        this.inventory = inventory;
    }

    public void selectProduct(String code) {
        throw new UnsupportedOperationException("TODO: select a product");
    }

    public void insertCoin(Coin coin) {
        throw new UnsupportedOperationException("TODO: insert a coin");
    }

    public List<Coin> dispense() {
        throw new UnsupportedOperationException("TODO: dispense the selected product");
    }

    public List<Coin> cancel() {
        throw new UnsupportedOperationException("TODO: cancel the current transaction");
    }

    public void restock(String code, int quantity) {
        throw new UnsupportedOperationException("TODO: restock a product");
    }

    public int insertedCents() {
        return insertedCents;
    }

    public Product selectedProduct() {
        return selectedProduct;
    }

    public VendingMachineState state() {
        return state;
    }
}

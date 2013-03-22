package net.electroland.eio;

public class Value {

    private int value;
    private boolean isSuspect;

    public Value(byte b){
        setValue(b);
    }
    public Value(short s){
        setValue(s);
    }
    public Value(int i){
        setValue(i);
    }

    public int getValue() {
        return value;
    }
    public boolean isSuspect() {
        return isSuspect;
    }

    public void setSuspect(boolean isSuspect) {
        this.isSuspect = isSuspect;
    }
    public void setValue(byte b){
        value = (int) b & 0xFF;
    }
    public void setValue(short s){
        value = (int)s;
    }
    public void setValue(int i){
        value = i;
    }
}
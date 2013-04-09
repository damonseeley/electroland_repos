package net.electroland.eio;

import java.util.HashMap;

import net.electroland.eio.filters.Filter;



public class Value {

    private int     filtered, raw;
    private boolean isSuspect;
    private HashMap<String, Integer>filteredValues = new HashMap<String, Integer>();
    
    public Value(byte b){
        setValue(b);
        raw = filtered;
    }
    public Value(short s){
        setValue(s);
        raw = filtered;
    }
    public Value(int i){
        setValue(i);
        raw = filtered;
    }
    public Value(String serialData){ // expects [192,122,0]
        String[] tokens = serialData.split(",|\\[|\\]");
        raw = new Integer(tokens[1]);
        filtered = new Integer(tokens[2]);
        isSuspect = new Integer(tokens[3]) == 1;
    }
    // [192,331,0]
    public void serialize(StringBuffer sb){
        sb.append('[');
        sb.append(raw).append(',');
        sb.append(filtered).append(',');
        sb.append(isSuspect ? 1 : 0);
        sb.append(']');
    }

    public int getFilteredValue(String filterId){
        return filteredValues.get(filterId);
    }
    public int getFilteredValue() {
        return filtered;
    }
    public int getRawValue() {
        return raw;
    }

    public boolean isSuspect() {
        return isSuspect;
    }
    public void setSuspect(boolean isSuspect) {
        this.isSuspect = isSuspect;
    }
    public void setValue(byte b){
        filtered = (int) b & 0xFF;
    }
    public void setValue(short s){
        filtered = (int)s;
    }
    public void setValue(int i){
        filtered = i;
    }
    public void setValue(Filter filter, int value){
        synchronized(filteredValues){
            filteredValues.put(filter.getId(), value);
        }
    }

    public String toString(){
        return "Value[value=" + filtered + ", isSuspect=" + isSuspect + "]";
    }
}
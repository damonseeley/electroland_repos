package net.electroland.utils.process;

import java.io.InputStream;

abstract public class ProcessItem {

    private InputStream is;

    public InputStream getInputStream(){
        return is;
    }

    protected void setInputStream(InputStream is){
        this.is = is;
    }

    abstract public int getPID();

    abstract String getName();
}
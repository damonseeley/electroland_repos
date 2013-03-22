package net.electroland.eio;


abstract public class OutputChannel extends Channel {

    abstract public void setValue(Object value);

    abstract public void write(Value value);
}
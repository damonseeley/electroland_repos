package net.electroland.eio;


abstract public class OutputChannel extends Channel {

    abstract public void write(Value value);
}
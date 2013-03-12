package net.electroland.eio.devices;


abstract public class OutputChannel extends Channel {

    abstract public void setValue(Object value);

    abstract public byte[] write(Object value);
}
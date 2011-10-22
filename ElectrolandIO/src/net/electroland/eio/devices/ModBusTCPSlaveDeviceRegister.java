package net.electroland.eio.devices;


public class ModBusTCPSlaveDeviceRegister {

    int startRef;
    int length;

    public ModBusTCPSlaveDeviceRegister(int startRef, int length)
    {
        this.startRef = startRef;
        this.length = length;
    }
}
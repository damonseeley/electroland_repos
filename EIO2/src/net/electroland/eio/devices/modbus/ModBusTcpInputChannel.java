package net.electroland.eio.devices.modbus;

import net.electroland.eio.InputChannel;

public class ModBusTcpInputChannel extends InputChannel {

    final static int BYTE  = 0;
    final static int SHORT = 1;
    final static int UNINT = 2;

    protected int type;
}
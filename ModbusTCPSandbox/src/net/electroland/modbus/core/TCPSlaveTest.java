package net.electroland.modbus.core;

import net.wimpi.modbus.Modbus;
import net.wimpi.modbus.ModbusCoupler;
import net.wimpi.modbus.net.ModbusTCPListener;
import net.wimpi.modbus.procimg.SimpleDigitalIn;
import net.wimpi.modbus.procimg.SimpleDigitalOut;
import net.wimpi.modbus.procimg.SimpleInputRegister;
import net.wimpi.modbus.procimg.SimpleProcessImage;
import net.wimpi.modbus.procimg.SimpleRegister;

public class TCPSlaveTest {

	public static void main(String[] args) {
		try {
			/* The important instances and variables */
			ModbusTCPListener listener = null;
			SimpleProcessImage spi = null;
			int port = Modbus.DEFAULT_PORT;

			//1. Set port number from commandline parameter
			if(args != null && args.length ==1) {
				//String[] args;
				port = Integer.parseInt(args[0]);
			}

			//2. Prepare a process image
			spi = new SimpleProcessImage();
			spi.addDigitalOut(new SimpleDigitalOut(true));
			spi.addDigitalOut(new SimpleDigitalOut(false));
			spi.addDigitalIn(new SimpleDigitalIn(false));
			spi.addDigitalIn(new SimpleDigitalIn(true));
			spi.addDigitalIn(new SimpleDigitalIn(false));
			spi.addDigitalIn(new SimpleDigitalIn(true));
			spi.addRegister(new SimpleRegister(251));
			spi.addInputRegister(new SimpleInputRegister(45));

			//3. Set the image on the coupler
			ModbusCoupler.getReference().setProcessImage(spi);
			ModbusCoupler.getReference().setMaster(false);
			ModbusCoupler.getReference().setUnitID(15);   

			//4. Create a listener with 3 threads in pool
			listener = new ModbusTCPListener(3);
			listener.setPort(port);
			listener.start();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}//main


}

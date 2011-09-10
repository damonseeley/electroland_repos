package net.electroland.modbus.core;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class DITest2  extends Thread {

	ModbusTCPMaster mtm1,mtm2;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place
	InputStreamReader isr;
	BufferedReader br;

	static Logger logger = Logger.getLogger(DITest2.class);
	
	public boolean[] sensors = new boolean[18];
	public double[] tripTimes = new double[sensors.length];

	public DITest2() {

		mtm1 = new ModbusTCPMaster("192.168.1.61");
		mtm2 = new ModbusTCPMaster("10.22.33.120");

		
		
		try {
			mtm1.connect();
			mtm2.connect();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		double startTime = System.currentTimeMillis();
		
		for (int i=0; i<sensors.length; i++) {
			sensors[i] = false;
			tripTimes[i] = startTime;
		}
		
		/////////////// THREAD STUFF
		framerate = 30;
		isRunning = true;
		timer = new Timer(framerate);
		start();
	}

	

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {
			try {

				/** JOANNA - work in here.  The readInputRegisters delivers the data, then we
				 * just have to parse it out
				 */
				InputRegister[] regs1 = mtm1.readInputRegisters(0, 1);
				InputRegister[] regs2 = mtm2.readInputRegisters(0, 1);

				printOutput(regs1[0].toBytes(), "phoenix" + regs1.length);
				printOutput(regs2[0].toBytes(), "beckoff" + regs2.length);
				System.out.println(regs1[0].toBytes().length);

			} catch (ModbusException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			//Thread ops
			//logger.info(timer.sleepTime);
			timer.block();
		}


	}

	public static void printOutput(byte[] bytes, String label)
	{
		BitVector bv = BitVector.createBitVector(bytes);
		
		for (int i=0; i < bv.size(); i++)
		{
			System.out.print(bv.getBit(i) ? '1' : '0');
			if ((i+1) % 8 == 0){
				System.out.print(' ');
			}
		}
		System.out.println("<-- us on " + label);
//		System.out.println(bv.toString() + " " + label);
//		System.out.println(Util.bytesToHex(bytes, bytes.length)  + " " + label);
		
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DITest2 dit = new DITest2();

	}

}

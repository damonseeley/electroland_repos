package net.electroland.modbus.core;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class DITest2  extends Thread {

	ModbusTCPMaster mtm;

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

		mtm = new ModbusTCPMaster("192.168.1.61");

		try {
			mtm.connect();
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

				/*
				isr = new InputStreamReader(System.in);
			    br = new BufferedReader(isr);

			    try {
					String s = br.readLine();
					System.out.println(trips);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				 */

				//System.out.println(mtm.readInputRegisters(0, 8).getClass());
				//System.out.println("Length = " + mtm.readInputRegisters(0, 8).length);

				/** JOANNA - work in here.  The readInputRegisters delivers the data, then we
				 * just have to parse it out
				 */
				InputRegister[] regs = mtm.readInputRegisters(0, 1);
				//BitVector bv1 = mtm.readCoils(0, 8);
				//System.out.println(bv1.toString());
				//BitVector discretes = mtm.readInputDiscretes(0, 1);

				//byte[] bts = regs[0].toBytes();

				BitVector bv = BitVector.createBitVector(regs[0].toBytes());
				
				for (int i=0; i < bv.size(); i++)
				{
					if (bv.getBit(i)){
						System.out.println("ON on channel " + i);
					}
				}
//				System.out.println();
//				System.out.println(bv.toString());
				//System.out.println(bv.toString().charAt(9));
				for (int i=0; i<sensors.length; i++){
				if (bv.toString().charAt(i) == "1".charAt(0)) {
					if (!sensors[i]) {
						tripTimes[i] = System.currentTimeMillis();
						sensors[i] = true;
//						logger.info("CH " + i + " ON");
					}
				}
				

				if (bv.toString().charAt(i) == "0".charAt(0)) {
					if (sensors[i]) {
						double duration = (System.currentTimeMillis() - tripTimes[i])/1000;
						logger.info("CH " + i + " OFF, duration: " + duration);
					}
					sensors[i] = false;
				}
				}


				for (InputRegister ir : regs){
					//System.out.println(ir.getValue());
					//for inputs 1-8, set high serially returns:
					// 1,2,4,8,16,32,64,128
					//System.out.println(Integer.toBinaryString(0x10000 | ir.getValue()).substring(1));

				}


				/** END parsing work
				 * 
				 */


			} catch (ModbusException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			//Thread ops
			//logger.info(timer.sleepTime);
			timer.block();
		}


	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DITest2 dit = new DITest2();

	}

}

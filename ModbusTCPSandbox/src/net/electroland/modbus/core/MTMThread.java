package net.electroland.modbus.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import net.wimpi.modbus.ModbusException;
import net.wimpi.modbus.facade.ModbusTCPMaster;
import net.wimpi.modbus.procimg.InputRegister;
import net.wimpi.modbus.util.BitVector;

import org.apache.log4j.Logger;

public class MTMThread  extends Thread {

	ModbusTCPMaster mtm;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place
	InputStreamReader isr;
	BufferedReader br;

	static Logger logger = Logger.getLogger(MTMThread.class);
	
	public boolean[] sensors = new boolean[18];
	public double[] tripTimes = new double[sensors.length];

	public MTMThread(String ip, int fr) {
		
		framerate = fr;

		mtm = new ModbusTCPMaster(ip);

		
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
		isRunning = true;
		timer = new Timer(framerate);
		start();
	}

	
	private double lastframe;
	private double execTime;
	private int cycle = 0;
	private int reportFreq = 100;
	private double[] execAvg = new double[reportFreq];
	
	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();
		lastframe = curTime;

		while (isRunning) {
			try {

				/** JOANNA - work in here.  The readInputRegisters delivers the data, then we
				 * just have to parse it out
				 */
				//InputRegister[] regs1 = mtm1.readInputRegisters(0, 1);
				InputRegister[] regs = mtm.readInputRegisters(0, 1);

				if (cycle >= reportFreq){
					double rateAvg = 0.0;
					
					for (int i=0; i< execAvg.length; i++){
						rateAvg += execAvg[i];
					}
					rateAvg  = rateAvg/execAvg.length;
					
					printOutput(regs[0].toBytes(), "phoenix" + regs.length);
					//System.out.println("Frame Exec avg " + rateAvg + " ms");
					cycle = 0;
				}


			} catch (ModbusException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			execAvg[cycle] = System.currentTimeMillis() - lastframe;
			lastframe = System.currentTimeMillis();
			cycle++;

			//Thread ops
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
//		System.out.println(bv.toString() + " " + label);
//		System.out.println(Util.bytesToHex(bytes, bytes.length)  + " " + label);
		
	}
	
	public void stopClean(){
		
		//isRunning = false;
		//mtm.disconnect();
	}

}

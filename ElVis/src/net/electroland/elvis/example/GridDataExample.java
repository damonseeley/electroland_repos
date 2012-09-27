package net.electroland.elvis.example;

import java.net.SocketException;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.PresenceGridUDPClient;

public class GridDataExample extends PresenceGridUDPClient {

	public GridDataExample(int port) throws SocketException {
		super(port);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void handle(GridData t) {
		System.out.println("--");
		for(int y =0; y < t.height; y++) {
			for(int x =0; x < t.width; x++) {
				
				System.out.print(t.getValue(x, y) + "  ");
			}
			System.out.println("");
		}


	}

	/**
	 * @param args
	 * @throws SocketException 
	 */
	public static void main(String[] args) throws SocketException {
		new GridDataExample(3458).start();

	}

}

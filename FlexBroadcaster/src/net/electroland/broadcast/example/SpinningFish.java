package net.electroland.broadcast.example;

import net.electroland.broadcast.server.FlexBroadcasterUtil;
import net.electroland.broadcast.server.XMLSocketBroadcaster;

public class SpinningFish {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int x = 100;
		int y = 100;		// center of circle
		int r = 100;			// radius

		int port = 1024;	// listener port
		final double TWOPI=2*Math.PI;

		
//		String id1 = FlexBroadcasterUtil.getUniqueId();	// fish ID
		String id2 = FlexBroadcasterUtil.getUniqueId();	// fish ID

		XMLSocketBroadcaster xmlsb = new XMLSocketBroadcaster(port);
		xmlsb.start();

		double a = 0;
		double b = 0;
					
		while(true){
			
			String header = "<pool>";
			
//			String xml1 = "<fish t=\"0\" id=\"" + id1 + "\" " + // id
////			"x=\"0\" y=\"0\" " +
//			"x=\"" + (int)(x + (r * Math.cos(a))) + "\" " + // x					
//			"y=\"" + (int)(y + (r * Math.sin(a))) + "\" " + // y
//			"o=\"" + (b) + "\" " + // o
//			"p=\"1.0\" />";

			String xml2 = "<fish t=\"0\" id=\"" + id2 + "\" " + // id
//			"x=\"0\" y=\"0\" " +
			"x=\"" + (int)(x + (r * Math.sin(a))) + "\" " + // x					
			"y=\"" + (int)(y + (r * Math.cos(a))) + "\" " + // y
			"o=\"" + (-1*(b++)) + "\" " + // o
			"p=\"1.0\" />";
			
			
			a = (a >= TWOPI) ? 0.0 : a + TWOPI / ( 360.0 );

			String footer = "</pool>";
						
			xmlsb.send(new TestXMLSocketMessage(header + xml2 + footer));

			try {
				Thread.sleep(33);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
		}
	}
}
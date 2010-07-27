using System;
using System.Collections.Generic;
using System.Text;

namespace Electroland
{


	public class GTrack
	{
		static int DEFAULT_PORT = 7878;
		StringBuilder sb = new StringBuilder();
		
		UDPSender sender;
		UDPReceiver receiver;
		GTrackParser gtrackParser;
		
		public GTrack () : this(DEFAULT_PORT)
		{
			
		}
		
		public GTrack(int port) {
			sender = new UDPSender(port);
			gtrackParser = new GTrackParser ();
			receiver = new UDPReceiver (port, gtrackParser);

		}
		
		public void close() {
			sender.close();
			receiver.close();
		}
		
		public void detectRegion(string id, float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
			sb.Length = 0; // clear
			sb.Append("D"); // D for detect
			sb.Append(id);
			sb.Append(":");
			sb.Append(minX); sb.Append(",");
			sb.Append(minY); sb.Append(",");
			sb.Append(minZ); sb.Append(",");
			sb.Append(maxX); sb.Append(",");
			sb.Append(maxY); sb.Append(",");
			sb.Append(maxZ); 
			sender.send(sb.ToString());
		}
		
		public void stopDetectRegion(string id) {
			sb.Length = 0; // clear
			sb.Append("F"); // f for free
			sb.Append(id);
			sender.send(sb.ToString());
		}
		public Dictionary<string, Dictionary<string, float>> getData() {
			return gtrackParser.get();
		}
	}
}


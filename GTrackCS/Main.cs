using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Collections.Generic;


namespace Electroland
{
	class MainClass
	{
		static int TEST_PORT = 7878;


		public static void Main (string[] args)
		{
			StringBuilder test = new StringBuilder();
			test.Append(1);
			Console.WriteLine(test.ToString());
			test.Append(2);
			Console.WriteLine(test.ToString());
			test.Length = 0;
			test.Append(1);
			Console.WriteLine(test.ToString());
			
			GTrackParser tsl = new GTrackParser ();
			UDPSender sender = new UDPSender(TEST_PORT);
			
			
			UDPReceiver r = new UDPReceiver (TEST_PORT, tsl);
			
			
			
			for (int i = 0; i < 10; i++) {
				string msg = "obj {" + "foo : 1;" + "bar : 2;" + "baz : 3;}, " + "obj2" + " {" + "foo:4.0;bar :5.0;baz: 6.0} ";
				sender.send (msg);
			}
			
			Thread.Sleep (1000);
			r.close ();
			sender.close();			
			
			foreach (KeyValuePair<string, Dictionary<string, float>> objEnt in tsl.get ()) {
				Console.WriteLine(objEnt.Key + "{");
				foreach (KeyValuePair<string, float> entry in objEnt.Value) {
					Console.WriteLine("  " + entry.Key + " : " + entry.Value + ";");
				}
				Console.WriteLine("}");
				
			}
		}
	}
}


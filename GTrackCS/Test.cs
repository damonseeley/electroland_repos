using System;
using System.Threading;
namespace Electroland
{
	public class Test
	{
       static void Main(string[] args)
  	  {
			GTrack gtrack = new GTrack();
			gtrack.send("foo { bar : 1 ; baz : 2; moof : 3}");
			Thread.Sleep(100);
			GTrack.prettyPrint(gtrack.getData());
			
    	}	
	}
}


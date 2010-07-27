using System;
using System.Threading;
using System.Text;

namespace Electroland
{
	public class ThreadedStringListener
	{
		ConcurrentLinkedList<string> cll;
		bool isRunning = true;

		public ThreadedStringListener ()
		{
			cll = new ConcurrentLinkedList<string> ();
		}

		public void start ()
		{
			new Thread (new ThreadStart (loop)).Start ();
		}

		public void loop ()
		{
			while (isRunning) {
				lock (this) {
					Monitor.Wait (this);
				}
				string s = cll.get ();
				while (s != null) {
					process (s);
					s = cll.get ();
				}
			}
		}

		public virtual void process (string s)
		{
			
		}

		public void put (string s)
		{
			cll.put (s);
			lock (this) {
				Monitor.Pulse (this);
			}
		}
		public void end ()
		{
			isRunning = false;
			lock (this) {
				Monitor.PulseAll (this);
			}
		}
	}
}


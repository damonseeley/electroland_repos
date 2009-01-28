package net.electroland.scSoundControl;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.util.concurrent.*;
import java.util.Date;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

import com.illposed.osc.*;

public class SCSoundControl implements OSCListener, Runnable {

	private OSCPortOut _sender;
	private OSCPortIn _receiver;
	
	//TODO: _motherGroupID perhaps should not be final. If there's a chance that two
	//computers could be controlling the same scsynth, then the motherGroupID
	//should be set pending a query of existing groups on the scsynth
	private final int _motherGroupID = 10000;
	
	private ConcurrentHashMap<Integer, String> _bufferMap;
	private Vector<Integer> _busList;
	private Vector<Integer> _nodeIdList;
	private int _minAudioBus, _minNodeID;
	private int _inChannels, _outChannels;
	private ConcurrentHashMap<Integer, SoundNode> _soundNodes; // keep track of
																// sound nodes
																// by their
																// group id.

	Thread _serverPingThread;
	public int _serverResponseTimeout = 200; //the max allowable time for server to respond to /status queries.
	public int _scsynthPingInterval = 100; //in milliseconds
	private Date _prevPingResponseTime, _prevPingRequestTime;
	
	SCSoundControlNotifiable _notifyListener;
	private boolean _serverLive;
	private boolean _serverBooted;
	
	private int _bufferReadIdOffset = 1000;
	
	private String[] _synthdefFile = 
		{"SuperColliderSynthDefs/ELenv.scsyndef", 
		"SuperColliderSynthDefs/ELplaybuf.scsyndef"};
	
	// if for some reason the system has more than 8 input channels, be sure to
	// specify that here.
	public SCSoundControl(int outputChannels, int inputChannels, SCSoundControlNotifiable listener) {

		// find the minimum id of a private audio bus.
		// by default, SC defines 8 channels out, and 8 more in.
		// so this will be at least 16, with 8 or fewer output channels
		_inChannels = inputChannels;
		_outChannels = outputChannels;
		_minAudioBus = Math.max(inputChannels, 8) + Math.max(outputChannels, 8);

		_minNodeID = 2; // SC allocates 0 as the root node and 1 as the default
						// group. The rest are up for grabs.

		// initialize id data structures
		_bufferMap = new ConcurrentHashMap<Integer, String>();
		_busList = new Vector<Integer>();
		_nodeIdList = new Vector<Integer>();
		_soundNodes = new ConcurrentHashMap<Integer, SoundNode>();

		// open the port for sending
		try {
			_sender = new OSCPortOut(InetAddress.getLocalHost(), OSCPort
					.defaultSCOSCPort());

		} catch (Exception e) {
			e.printStackTrace();
		}

		// begin by listening for messages:
		try {
			_receiver = new OSCPortIn(_sender.getSocket());
			_receiver.addListener(".*", this); // receive all notify info
			_receiver.startListening();
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		_prevPingResponseTime =  _prevPingRequestTime = new Date();
		_notifyListener = listener;
		_serverLive = false;
		_serverBooted = false;
		
		//start the thread that will ping scsynth
		_serverPingThread = new Thread(this);
		_serverPingThread.setPriority(Thread.MIN_PRIORITY);
		_serverPingThread.start();
		
		init();

	}

	// cleanup if we're getting gc'ed.
	protected void finalize() throws Throwable {
		cleanup();

		// free the UDP port from JavaOSC
		if (_receiver != null)
			_receiver.close();

	}

	// establish group to contain all SCSoundControl nodes in SuperCollider
	public void init() {
		
		// can sanity check some things by requesting notification when nodes
		// are created/deleted/etc.
		// It does not notify when playBufs reach the end of a buffer, though.
		// Too bad, that. Would have to poll.
		sendMessage("/notify", new Object[] { 1 });

		//start by cleaning up any detritus from previous runs on the same server:
		cleanup();
		
		// create a mother group, under the default group (1),
		// which will contain all of the SCSoundControl objects.
		
		//this is where we would query if another node == _motherGroupID already exists.
		//if so, would need to choose an alternate groupID (e.g. += 10000)
		createGroup(_motherGroupID, 1);
		
		//load the synthdefs onto the server - not working yet.
//		for (int i=0; i < _synthdefFile.length; i++) {
//			File f = new File(_synthdefFile[i]);
//			if (f.canRead()) {
//				FileInputStream stream = new FileInputStream(f);
//			}
//			char[] synthdata = new char[0];
//		
//			sendMessage("/d_recv", new Object[]{synthdata});
//		}

		
	}

	// cleanup all SCSoundControl nodes in SuperCollider
	public void cleanup() {
		// by freeing the mother group we clean up all the nodes we've created
		// on the SC server
		sendMessage("/g_freeAll", new Object[] { _motherGroupID });
		freeNode(_motherGroupID);
		freeAllBuffers();
		
		//reset the lists of sound nodes, nodeIds, busses, etc
		SoundNode sn;
		Enumeration<SoundNode> e = _soundNodes.elements();
		while (e.hasMoreElements()) {
			sn = e.nextElement();
			sn.setAlive(false);
		}
		_soundNodes.clear();
		_nodeIdList.clear();
		_busList.clear();
		_bufferMap.clear();
		
	}

	//***************
	//It is up to the user to be sure that the filename exists on the machine running SuperCollider.
	//***************
	// by default, SuperCollider provides 1024 buffers.
	//
	//Ideally, this would register a callback to the requesting object
	//and then notify when the buffer was done loading. But super collider
	//only notifies that a buffer finished loading. Not which one.
	//In the case you load one HUGE buffer and immediately load another tiny one,
	//the second might finish before the first, and you wouldn't know which one was
	//valid - or which one threw an error...
	public int readBuf(String filename) {
		
		// find a buffer id that hasn't been used yet.
		int bufNum = 0;
		while (_bufferMap.containsKey(bufNum))
			bufNum++;

		// add this buffer number to the map
		_bufferMap.put(bufNum, filename);

		// create and load the buffer
		sendMessage("/b_allocRead", new Object[] { bufNum, filename });
		sendMessage("/sync", new Object[]{bufNum + _bufferReadIdOffset});
		return bufNum;
	}

	// free a given buffer from the SC server.
	public void freeBuf(int bufNum) {
		// if it's a valid remove request:
		if (_bufferMap.containsKey(bufNum)) {
			// remove it from the buffer map
			_bufferMap.remove(bufNum);
			// free it on the server
			sendMessage("/b_free", new Object[] { bufNum });
		}
	}

	// free all the buffers we've allocated on the SC server.
	public void freeAllBuffers() {
		Iterator<Integer> iter = _bufferMap.keySet().iterator();
		while (iter.hasNext()) {
			int bufNum = iter.next();
			sendMessage("/b_free", new Object[] { bufNum });
			iter.remove();
		}
	}

	// by default, SuperCollider allocates 128 audio rate busses.
	// The way SCSoundControl has been designed, this should allow for 128 voice
	// polyphony.
	// If more are needed, see super collider help on how to do this from
	// either command line arguments or by changing defaults via sclang code.
	protected synchronized int createBus() {
		int newBusID = _minAudioBus;
		while (_busList.contains(newBusID))
			newBusID++;
		_busList.add(newBusID);
		return newBusID;
	}

	// free up a no longer used bus id
	protected synchronized void freeBus(int busNum) {
		_busList.remove(new Integer(busNum));
	}

	// get an unallocated node id.
	protected int getNewNodeID() {
		int newID = _minNodeID;
		while (_nodeIdList.contains(newID)) newID++;
		return newID;
	}

	// free a node we've allocated on the server.
	protected void freeNode(int nodeNum) {
		sendMessage("/n_free", new Object[] { nodeNum });
	}

	// autogenerate a new group and return the group ID.
	// groups all have ids higher than the SoundControl "mothergroup"
	protected int createGroup() {
		int id = _motherGroupID + 1;
		while (_soundNodes.containsKey(id))
			id++;
		createGroup(id);
		return id;
	}

	// create a group under the SCSoundControl mothergroup
	protected void createGroup(int id) {
		createGroup(id, _motherGroupID);
	}

	// create a group on the server.
	protected void createGroup(int id, int parentGroupID) {
		sendMessage("/g_new", new Object[] { id, 0, parentGroupID });

	}

	// create an ELplaybuf node. Note this is a custom synthdef which must be
	// loaded on the server.
	protected void createPlayBuf(int id, int group, int bufNum, int outBus,
			float amp, boolean loop) {

		Object args[] = new Object[14];
		args[0] = new String("ELplaybuf");
		args[1] = new Integer(id); // need a unique ID
		args[2] = new Integer(1); // add to tail of node list in group
		args[3] = new Integer(group); // target group
		args[4] = new String("outBus");
		args[5] = new Integer(outBus); // need a unique bus # here
		args[6] = new String("bufNum");
		args[7] = new Integer(bufNum);
		args[8] = new String("doLoop");
		args[9] = loop ? new Integer(1) : new Integer(0);
		args[10] = new String("ampScale");
		args[11] = new Float(amp);
		args[12] = new String("groupToFreeWhenDone");
		args[13] = new Integer(group);

		sendMessage("/s_new", args);
		_nodeIdList.add(id);
	}

	// create an ELenv node. Note this is a custom synthdef which must be loaded
	// on the server.
	protected void createEnvelope(int id, int group, int inBus, int outBus,
			float amp) {

		Object args[] = new Object[12];
		args[0] = new String("ELenv");
		args[1] = new Integer(id); // need a unique ID
		args[2] = new Integer(1); // add to tail of group
		args[3] = new Integer(group);
		args[4] = new String("inBus");
		args[5] = new Integer(inBus); // need a unique bus # here
		args[6] = new String("outChannel");
		args[7] = new Integer(outBus);
		args[8] = new String("ampScale");
		args[9] = new Float(amp);

		sendMessage("/s_new", args);

		_nodeIdList.add(id);
	}

	// set the amplitude of an ELplaybuf or an ELenv.
	public void setAmplitude(int id, float amp) {
		sendMessage("/n_set", new Object[] { id, "ampScale", amp });
	}

	// helper function
	protected void sendMessage(String addr) {
		sendMessage(addr, null);
	}

	// helper function, we use this all over the place.
	protected void sendMessage(String addr, Object[] args) {
		try {
			_sender.send(args == null ? new OSCMessage(addr) : new OSCMessage(
					addr, args));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// reset the SC server - same as pressing "CMD-." in sclang
	// just for development. Shouldn't need to call this typically.
	// it would kill nodes created by something else controlling the server.
	public void reset() {
		sendMessage("/g_freeAll", new Object[] { 0 });
		sendMessage("/clearSched");
		sendMessage("/g_new", new Object[] { 1 });
	}

	// for debugging: tell scsynth to dump a tree of its nodes to the post
	// windows
	public void dumpTree() {
		sendMessage("/g_dumpTree", new Object[] { 0, 0 });
	}

	// for debugging: tell scsynth to trace a given node in the post window
	public void trace(int node) {
		sendMessage("/n_trace", new Object[] { node });
	}

	// handle incoming messages
	public void acceptMessage(java.util.Date time, OSCMessage message) {
		// to print the full message:
		if (false) {
		 debugPrint(message.getAddress());
		 for(int i = 0; i < message.getArguments().length; i++) {
			 debugPrint(" " + message.getArguments()[i].toString());
		 }
		 debugPrintln("");
		}
		
		if (message.getAddress().matches("/done")) {
			if (message.getArguments()[0].toString().matches("/b_allocRead")) {
				//debugPrintln("A buffer was created.");
			}
		}

		else if (message.getAddress().matches("/synced")) {
			int id = (Integer)(message.getArguments()[0]) - _bufferReadIdOffset;
			if (_bufferMap.containsKey(id)) {
				debugPrintln("Buffer " + id + " was loaded.");
				_notifyListener.receiveNotification_BufferLoaded(id, _bufferMap.get(id));
			}
		}
		
		else if (message.getAddress().matches("/n_go") ||
			message.getAddress().matches("/n_info")) {
			//if it was node 1, then we can get going.
			if ((Integer)(message.getArguments()[0]) == 1 && 
				(Integer)(message.getArguments()[1]) == 0 ) {
				if (!_serverBooted) {
					_serverBooted = true;
					handleServerBooted();
				}
			}
		}

		// handle notices of freed nodes
		else if (message.getAddress().matches("/n_end")) {
			// take message.getArguments()[0] (the node id that was freed)
			// and free up any resources associated with it.
			debugPrintln("node " + message.getArguments()[0].toString()
					+ " was freed.");
			Integer id = (Integer) (message.getArguments()[0]);
			if (id > _motherGroupID) {
				// it's a group node. a SoundNode has died.
				synchronized (this) {
					SoundNode sn = _soundNodes.get(id);
					if (sn != null) {
						sn.setAlive(false);
						freeBus(sn.get_bus());
					}
					_soundNodes.remove(id);
				}
			} else {
				// free up that node from the node id list.
				synchronized (this) {
					_nodeIdList.remove(message.getArguments()[0]);
				}
			}
		}
		
		//handle /status responses (status.reply)
		else if (message.getAddress().matches("status.*")) {
			Float avgCPU = (Float)(message.getArguments()[5]);
			Float peakCPU = (Float)(message.getArguments()[6]);
			handleServerStatusUpdate(avgCPU, peakCPU);
		}
		
	}

	public SoundNode createSoundNodeOnSingleChannel(int bufferNumber, boolean doLoop, int channel, float amplitude) {
		float[] amps = new float[_outChannels];
		for (int i=0; i < _outChannels; i++) {
			amps[i] = i==channel? amplitude : 0;
		}
		return createSoundNode(bufferNumber, doLoop, amps);
	}
	
	// create a new soundNode to playback a buffer
	public SoundNode createSoundNode(int bufferNumber, boolean doLoop,
			float[] channelAmplitudes) {
		
		//sanity check the buffer that's been requested.
		if (!_bufferMap.containsKey(bufferNumber)) return null;
		
		// establish group and node id's for this sound node.
		int newGroup = createGroup();

		// instantiate a new SoundNode and remember it by its group ID.
		// when that group ID is freed on the server we'll know this SoundNode
		// is dead.
		SoundNode sn;
		synchronized (this) {
			sn = new SoundNode(this, newGroup, bufferNumber, doLoop,
					_outChannels, channelAmplitudes);
			_soundNodes.put(newGroup, sn);			
		}

		return sn;
	}
	

	// debug output helpers
	boolean _doDebug = false;

	public void showDebugOutput(boolean state) {
		_doDebug = state;
	}

	public void debugPrint(String s) {
		if (_doDebug)
			System.out.print(s);
	}

	public void debugPrintln(String s) {
		if (_doDebug)
			System.out.println(s);
	}

	protected void handleServerBooted() {
		//reinit data.
		this.init();
		//notify client.
		_notifyListener.receiveNotification_ServerRunning();		
	}
	
	protected void handleServerStatusUpdate(float avgCPU, float peakCPU) {
		_prevPingResponseTime = new Date();
		
		//if (!_serverLive || !_serverBooted) {
		if (!_serverLive) {
			_serverLive = true;
			//Server is running. Query for node 1 (the sign that server's booted).
			sendMessage("/notify", new Object[] { 1 });
			sendMessage("/n_query", new Object[]{1});	
			debugPrintln("Querying node 1");
		}
		
		_notifyListener.receiveNotification_ServerStatus(avgCPU, peakCPU);
		
		//debugPrintln("status latency: " + (_prevPingResponseTime.getTime() - _prevPingRequestTime.getTime()));
	}
	
	
	//SCSoundControl starts up a thread to make sure the server is running.
	public void run() {
		Date curTime;
		while(true) {
			curTime = new Date();

			//if the previous status request is still pending...
			if (_serverLive && (_prevPingRequestTime.getTime() > _prevPingResponseTime.getTime())) {
				//We've not yet heard back from the previous status request.
				//Have we timed out?
				if (curTime.getTime() - _prevPingRequestTime.getTime() > _serverResponseTimeout) {
					//We've timed out on the previous status request.
					_notifyListener.receiveNotification_ServerStopped();
					_serverLive = false;
					_serverBooted = false;
				}
				//else we just keep waiting for a response or a timeout
			}
			//the previous status request is NOT still pending. Is it time to send another?
			else if (curTime.getTime() - _prevPingRequestTime.getTime() > _scsynthPingInterval) {
				//It's time to send another status request.
				sendMessage("/status");
				_prevPingRequestTime = new Date();
			}
			//it's not time to send, and we're not watching 
			//for a reply, so go to sleep until it's time to ping again
			else {
				try {
					Thread.sleep(_scsynthPingInterval - (curTime.getTime() - _prevPingRequestTime.getTime()));
				} catch (InterruptedException e) {
					//NOTE this thread shouldn't get interrupted.
					e.printStackTrace();
				}			
			}
			
			
		}
	}

	
	//modify server ping settings: ************************************
	public int get_serverResponseTimeout() {
		return _serverResponseTimeout;
	}
	public void set_serverResponseTimeout(int responseTimeout) {
		_serverResponseTimeout = responseTimeout;
	}
	public int get_scsynthPingInterval() {
		return _scsynthPingInterval;
	}
	public void set_scsynthPingInterval(int pingInterval) {
		_scsynthPingInterval = pingInterval;
	}
	//*****************************************************************
	

}


package net.electroland.scSoundControl;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.util.concurrent.*;
import java.util.Date;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Properties;
import java.util.Vector;

import javax.swing.JFrame;

import org.apache.log4j.Logger;

import com.illposed.osc_ELmod.*;

public class SCSoundControl implements OSCListener, Runnable {

	/**
	 * SCSoundControl provides various services:
	 * It launches scsynth, the super collider server.
	 * It handles messaging with scsynth via OSC.
	 * It provides a simple public interface for playing and controlling sounds via scsynth.
	 * 
	 * TODO: USAGE EXAMPLES
	 */
	
	//ports for sending and receiving OSC messages:
	private OSCPortOut _sender;
	private OSCPortIn _receiver;
	
	//the "mother group", a group to contain all the other group nodes. 
	private final int _motherGroupID = 10000;
	
	//Data Structures
	private ConcurrentHashMap<Integer, String> _bufferMap;
	private Vector<Integer> _busList;
	private Vector<Integer> _nodeIdList;
	private int _minAudioBus, _minNodeID;
	private int _inChannels, _outChannels;
	//sound nodes are keyed off their group ID
	private ConcurrentHashMap<Integer, SoundNode> _soundNodes; 
	
	//we monitor the state of the server by "pinging" it on a thread.
	private Thread _serverPingThread;
	// FIXME increasing _serverResponseTimeout causes less disconnections caused by (I think) a high number of synths.
	private int _serverResponseTimeout = 2000; //200; //the max allowable time for server to respond to /status queries.
	private int _scsynthPingInterval = 100; //in milliseconds
	private Date _prevPingResponseTime, _prevPingRequestTime;

	//the state of the scsynth server.
	private boolean _serverLive;
	private boolean _serverBooted;
	
	// set this to true for more verbose output
	private boolean debugging = false;

	//the client code, which will receive notifications of scsynth status and events
	private SCSoundControlNotifiable _notifyListener;
	
	//used for internal bookkeeping...
	//when a buffer read request is completed by scsynth, it replies with
	//a number (specified in the original request). We know what buffer has finished loading
	//by adding the buffer id number to this value:
	//i.e. when a reply of value 1005 is receieved, we know that buffer 5 finished.
	//this same technique can be used to confirm completion of other asynchronous events (but using a different offset)
	private int _bufferReadIdOffset = 1000;
	
	//a pointer to the scsynth OS process
	private ScsynthLauncher _scsynthLauncher;
	
	//the gui control panel
	private SCSoundControlPanel _controlPanel;
	
	//load properties from a file
	private Properties _props;
	private String _propertiesFilename;	
	//the max polyphony, set in the properties file:
	private int _maxPolyphony = 64;
	
	static Logger logger = Logger.getLogger(SCSoundControl.class);
	
	
	/**
	 * Create an instance of SCSoundControl, using the default properties file.
	 * @param listener an object that will receive notifications from this instance of SCSoundControl
	 */
	//providing default parameters:
	public SCSoundControl(SCSoundControlNotifiable listener) {
		//this defines the default properties filename:
		this(listener, "depends/SCSoundControl.properties");
	}
	
	/**
	 * Create an instance of SCSoundControl
	 * 
	 * @param listener an object that will receive notifications from this instance of SCSoundControl 
	 * @param propertiesFilename provide a unique filename for the properties file.
	 * This allows multiple use scenarios (e.g. testing different audio hardware)
	 */
	public SCSoundControl(SCSoundControlNotifiable listener, String propertiesFilename) {
		_propertiesFilename = propertiesFilename;
		showDebugOutput(false);

		//load properties from a file
		loadPropertiesFile(_propertiesFilename);
		
		//setup synth launcher so that scsynth properties are handled and defaults are setup as needed.
        _scsynthLauncher = new ScsynthLauncher(this, _props);
		_maxPolyphony = _scsynthLauncher.getMaxPolyphony();
		//TODO: use maxPolyphony to check before new soundNode creation...

		// find the minimum id of a private audio bus.
		// if certain properties are unset, use the scsynth defaults
		_inChannels = Integer.valueOf(_props.getProperty("SuperCollider_NumberOfInputChannels", "8"));
		_outChannels = Integer.valueOf(_props.getProperty("SuperCollider_NumberOfOutputChannels", "8"));
		_minAudioBus = _inChannels + _outChannels;

		_minNodeID = 2; //We're following the scsynth convention of allocating 0 as the 
						//root node and 1 as the default group. The rest are up for grabs.

		// initialize id data structures
		_bufferMap = new ConcurrentHashMap<Integer, String>();
		_busList = new Vector<Integer>();
		_nodeIdList = new Vector<Integer>();
		_soundNodes = new ConcurrentHashMap<Integer, SoundNode>();

		//grab the scsynth udp port for the socket:
		int udpport = Integer.valueOf(_props.getProperty("SuperCollider_UDPportNumber", "57110"));
		
		// open the port for sending
		try { _sender = new OSCPortOut(InetAddress.getLocalHost(), udpport); }
		catch (Exception e) { e.printStackTrace();}

		// begin by listening for messages:
		try {
			_receiver = new OSCPortIn(_sender.getSocket());
			_receiver.addListener(".*", this); // receive all notify info
			_receiver.startListening();
		} catch (Exception e1) { e1.printStackTrace(); }

		//state variables
		_prevPingResponseTime =  _prevPingRequestTime = new Date();
		_notifyListener = listener;
		_serverLive = false;
		_serverBooted = false;
				
		//create the GUI: the control panel
		_controlPanel = new SCSoundControlPanel();
        _controlPanel.setOpaque(true); //content panes must be opaque

        //TODO: maybe we don't want to create a frame, but rather hand off the panel.
		//setup a window to display the control panel
        JFrame frame = new JFrame("SuperColliderSoundControl");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setContentPane(_controlPanel);
        frame.pack();
        frame.setSize(400, 300);
        frame.setVisible(true);
        
		//start the server.
		bootScsynth();
        
		//start the thread that will ping scsynth
		_serverPingThread = new Thread(this);
		_serverPingThread.setPriority(Thread.MIN_PRIORITY);
		_serverPingThread.start();

	}

	/**
	 * Load properties from a file.
	 * @param propertiesFile a string referring to a properties file. If not found, it will be created if possible.
	 * 
	 */
	private void loadPropertiesFile(String propertiesFile) {

		//open the properties file.
		_props = new Properties();

		//sanity check the properties file
		File f = new File(propertiesFile);
		if (!f.canRead()) {
			//print an error - can't read the props file.
			System.err.println("Properties file " + propertiesFile + " cannot be read.");
		}

		try {
			_props.load(new FileInputStream(f));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Output the current state of SCSoundControl properties to a file.
	 * @param propertiesFile the output filename
	 */
	private void savePropertiesFile(String propertiesFile) {
		if (_props == null) return; //or maybe save a default file?
		File f = new File(propertiesFile);
		
		//does it exist yet?
		if (!f.exists()) {
			//touch the file and close it.
			try {f.createNewFile();}
			catch (Exception e) {
				System.err.println("Properties file " + propertiesFile + " does not exist and cannot be created.");
			}
		}
		
		//write the file.
		try {
			_props.store(new FileOutputStream(f), "SCSoundControl Properties File");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

	
	// cleanup if we're getting gc'ed.
	protected void finalize() throws Throwable {
		cleanup();

		//kill the scsynth
		_scsynthLauncher.killScsynth();
		
		// free the UDP port from JavaOSC
		if (_receiver != null)
			_receiver.close();

	}

	// establish group to contain all SCSoundControl nodes in SuperCollider
	public void init() {
		
		debugPrintln("Doing init.");
		
		// can sanity check some things by requesting notification when nodes
		// are created/deleted/etc.
		// It does not notify when playBufs reach the end of a buffer, though.
		// Too bad, that. Would have to poll.
		sendMessage("/notify", new Object[] { 1 });

		//start by cleaning up any detritus from previous runs on the same server:
		cleanup();

		//sclang creates the default group, not supercollider, so let's follow that convention.
		createGroup(1, 0);

		// create a mother group, under the default group (1),
		// which will contain all of the SCSoundControl objects.
		//this is where we would query if another node == _motherGroupID already exists.
		//if so, would need to choose an alternate groupID (e.g. += 10000)
		createGroup(_motherGroupID, 1);
		
	}

	public void shutdown() {
		cleanup();
		savePropertiesFile(_propertiesFilename);
		quitScsynth();
		//TODO if the quit message fails, need to kill scsynth
		//_scsynthLauncher.killScsynth();
	}
	
	/**
	 *  cleanup all SCSoundControl nodes in SuperCollider
	 */
	public synchronized void cleanup() {
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
	
	private void bootScsynth() {
		
		_scsynthLauncher.launch();
		_controlPanel.connectScsynthOutput(_scsynthLauncher.getScsynthOutput());

	}
	

	//*********************************
	// Buffer Management
	//*********************************
	
	// It is up to the user to be sure that the filename exists on the machine running SuperCollider.
	// by default, SuperCollider provides 1024 buffers.
	
	public synchronized int readBuf(String filename) {
		
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
	public synchronized void freeBuf(int bufNum) {
		// if it's a valid remove request:
		if (_bufferMap.containsKey(bufNum)) {
			// remove it from the buffer map
			_bufferMap.remove(bufNum);
			// free it on the server
			sendMessage("/b_free", new Object[] { bufNum });
		}
	}

	// free all the buffers we've allocated on the SC server.
	public synchronized void freeAllBuffers() {
		Iterator<Integer> iter = _bufferMap.keySet().iterator();
		while (iter.hasNext()) {
			int bufNum = iter.next();
			sendMessage("/b_free", new Object[] { bufNum });
			iter.remove();
		}
	}

	
	//*********************************
	// Internal Bus ID Management
	//*********************************
	
	// by default, SuperCollider allocates 128 audio rate busses.
	// The way SCSoundControl has been designed, this should allow for 128 voice
	// polyphony.
	// If more are needed, see super collider help on how to do this from
	// either command line arguments or by changing defaults via sclang code.
	
	protected synchronized int allocateBus() {
		int newBusID = _minAudioBus;
		while (_busList.contains(newBusID))
			newBusID++;
		_busList.add(newBusID);
		return newBusID;
	}
	
	//find consecutive busses, return lowest id.
	protected synchronized int allocateConsecutiveBusses(int howManyBusses) {
		int newBusID = _minAudioBus;
		while (!testConsecutiveBusses_recursive(newBusID, howManyBusses)) {
			newBusID++;
		}
		for (int i=0; i<howManyBusses; i++) {
			_busList.add(newBusID + i);
		}
		return newBusID;
	}
	
	//recursively look for consecutive bus ID's to be free.
	private boolean testConsecutiveBusses_recursive(int startID, int howMany) {
		if (howMany == 0) return true;
		if (_busList.contains(startID)) return false;
		else return testConsecutiveBusses_recursive(startID+1, howMany-1);
	}

	// free up a no longer used bus id
	protected synchronized void freeBus(int busNum) {
		_busList.remove(new Integer(busNum));
	}
	
	protected void freeConsecutiveBusses(int busNum, int howManyBusses) {
		for (int i=0; i < howManyBusses; i++) freeBus(busNum + i);
	}

	
	//*********************************
	// Node ID Management
	//*********************************
	
	// get an unallocated node id.
	public int getNewNodeID() {
		int newID = _minNodeID;
		while (_nodeIdList.contains(newID)) newID++;
		return newID;
	}

	//****************************************
	// Messaging to Super Collider
	//****************************************
	
	// free a node we've allocated on the server.
	protected void freeNode(int nodeNum) {
		sendMessage("/n_free", new Object[] { nodeNum });
	}

	// autogenerate a new group and return the group ID.
	// groups all have ids higher than the SoundControl "mothergroup"
	protected int createGroup() {
		int id = _motherGroupID + 1;
		while (_soundNodes.containsKey(id)) { id++; }
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
			float amp, float rate, boolean loop) {

		Object args[] = new Object[16];
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
		args[12] = new String("playbackRate");
		args[13] = new Float(rate);
		args[14] = new String("groupToFreeWhenDone");
		args[15] = new Integer(group);

		sendMessage("/s_new", args);
		_nodeIdList.add(id);
	}

	// create an ELplaybuf node. Note this is a custom synthdef which must be
	// loaded on the server.
	protected void createStereoPlayBuf(int id, int group, int bufNum, int outBus,
			float amp, float rate, boolean loop) {

		Object args[] = new Object[16];
		args[0] = new String("ELStereoPlaybuf");
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
		args[12] = new String("playbackRate");
		args[13] = new Float(rate);
		args[14] = new String("groupToFreeWhenDone");
		args[15] = new Integer(group);

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
	public void setProperty(int id, String paramName, float amp) {
		sendMessage("/n_set", new Object[] { id, paramName, amp });
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
	public void resetScsynth() {
		sendMessage("/g_freeAll", new Object[] { 0 });
		sendMessage("/clearSched");
		sendMessage("/g_new", new Object[] { 1 });
	}

	/**
	 * for debugging: tell scsynth to dump a tree of its nodes to the post windows
	 */
	public void dumpTree() {
		sendMessage("/g_dumpTree", new Object[] { 0, 0 });
	}

	/**
	 * for debugging: tell scsynth to trace a given node in the post window
	 * @param node: ID
	 */
	public void trace(int node) {
		sendMessage("/n_trace", new Object[] { node });
	}

	/**
	 * Quit ScSynth
	 */
	public void quitScsynth() {
		sendMessage("quit");
	}
	
	
	//**************************************
	// Handle Incoming messages from Super Collider
	//**************************************
	
	// handle incoming messages
	public void acceptMessage(java.util.Date time, OSCMessage message) {
		// FOR DEBUGGING: to print the full message:
		if (debugging) {
			logger.debug(message.getAddress());
			for(int i = 0; i < message.getArguments().length; i++) {
				logger.debug(", " + message.getArguments()[i].toString());
			}
			logger.debug("");
		}
		
		if (message.getAddress().matches("/done")) {
			if (message.getArguments()[0].toString().matches("/b_allocRead")) {
				//debugPrintln("A buffer was created.");
			}
		}

		//TODO: this /synced watch will work as long as the buffers load correctly.
		//but there's not yet any way to know if they fail.
		else if (message.getAddress().matches("/synced")) {
			int id = (Integer)(message.getArguments()[0]) - _bufferReadIdOffset;
			synchronized (this) {
				if (_bufferMap.containsKey(id)) {
					debugPrintln("Buffer " + id + " was loaded.");
					if (_notifyListener != null) { _notifyListener.receiveNotification_BufferLoaded(id, _bufferMap.get(id)); }
					if (_controlPanel != null && _controlPanel._statsDisplay != null) {
						_controlPanel._statsDisplay.receiveNotification_BufferLoaded(id, _bufferMap.get(id));
					}

				}
			}
		}
		
		else if (message.getAddress().matches("/n_go") ||
			message.getAddress().matches("/n_info")) {
			//if it was node 1, then we can get going.
			if ((Integer)(message.getArguments()[0]) == _motherGroupID && 
				(Integer)(message.getArguments()[1]) == 1 ) {
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
			//debugPrintln("node " + message.getArguments()[0].toString() + " was freed.");
			Integer id = (Integer) (message.getArguments()[0]);
			if (id > _motherGroupID) {
				// it's a group node. a SoundNode has died.
				synchronized (this) {
					SoundNode sn = _soundNodes.get(id);
					if (sn != null) {
						sn.setAlive(false);
						freeBus(sn.get_busID());
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
		else if (message.getAddress().matches("/status.*")) {
			//TODO update control panel display with status data
			Integer numUgens = (Integer)(message.getArguments()[1]);
			Integer numSynths = (Integer)(message.getArguments()[2]);
			Integer numGroups = (Integer)(message.getArguments()[3]);
			Integer numSynthdefs = (Integer)(message.getArguments()[4]);
			Float avgCPU = (Float)(message.getArguments()[5]);
			Float peakCPU = (Float)(message.getArguments()[6]);
			handleServerStatusUpdate(numUgens, numSynths, numGroups, numSynthdefs, avgCPU, peakCPU);
		}
		
	}


	//*****************************************
	// Sound Node Factory Methods
	//*****************************************

	/**
	 * Convenience function to create a new soundNode to playback a mono buffer 
	 * to a single output channel (can be used to play the left channel of a stereo buffer). Calls createMonoSoundNode().
	 * @param bufferNumber : id of the playback buffer
	 * @param doLoop : boolean - loop? or play then stop & die
	 * @param channel : which output channel the playback buffer should play to
	 * @param amplitude : the amplitude to begin playing at.
	 * @param playbackRate : amplitudes corresponding to the rChannelOutputChannels param.
	 * @return a new SoundNode
	 */
	public SoundNode createSoundNodeOnSingleChannel(int bufferNumber, boolean doLoop, int channel, float amplitude, float playbackRate) {
		int[] channels = new int[1];
		channels[0] = channel;
		
		float[] amps = new float[1];
		amps[0] = amplitude;
		
		return createMonoSoundNode(bufferNumber, doLoop, channels, amps, playbackRate);
	}
	
	/**
	 * create a new soundNode to playback a mono buffer (can be used to play the left channel of a stereo buffer)
	 * @param bufferNumber : id of the playback buffer
	 * @param doLoop : boolean - loop? or play then stop & die
	 * @param outputChannels : an array of output channels which the playback buffer should play to
	 * @param channelAmplitudes : amplitudes corresponding to the outputChannels param.
	 * @param playbackRate : amplitudes corresponding to the rChannelOutputChannels param.
	 * @return a new SoundNode
	 */
	public SoundNode createMonoSoundNode(int bufferNumber, boolean doLoop,
			int[] outputChannels, float[] channelAmplitudes, float playbackRate) {
		
		//sanity check the buffer that's been requested.
		if (!_bufferMap.containsKey(bufferNumber)) return null;
		
		//wrap up 1D input arrays into 2D arrays
		float[][] amps = new float[1][];
		amps[0] = channelAmplitudes; //this is ok, since the the amps array gets copied inside of SoundNode constructor.
		
		int[][] outChannels = new int[1][];
		outChannels[0] = outputChannels;
		
		// instantiate a new SoundNode and remember it by its group ID.
		// when that group ID is freed on the server we'll know this SoundNode
		// is dead.
		SoundNode sn;
		synchronized (this) {
			sn = new SoundNode(this, bufferNumber, 1, doLoop,
					outChannels, amps, playbackRate);
			_soundNodes.put(sn.getGroup(), sn);
		}

		return sn;
	}


	/**
	 * The use of this function is deprecated, as its functionality is made redundant by updates to createStereoSoundNode().
	 */
	@Deprecated public SoundNode createStereoSoundNodeWithLRMap(int bufferNumber, boolean doLoop,
			int[] leftChannelMap, int[] rightChannelMap, float playbackRate) {
		float[] lChannelAmplitudes = new float[leftChannelMap.length];
		float[] rChannelAmplitudes = new float[rightChannelMap.length];
		for (int i=0; i<leftChannelMap.length; i++) {
			lChannelAmplitudes[leftChannelMap[i]] = 1f;
		}
		for (int i=0; i<rightChannelMap.length; i++) {
			rChannelAmplitudes[rightChannelMap[i]] = 1f;
		}
		return createStereoSoundNode(bufferNumber, doLoop, leftChannelMap, lChannelAmplitudes, rightChannelMap, rChannelAmplitudes, playbackRate);
	}
	
	/**
	 * create a new soundNode to playback a stereo buffer
	 * @param bufferNumber : id of the playback buffer
	 * @param doLoop : boolean - loop? or play then stop & die
	 * @param lChannelOutputChannels : an array of output channels which the left channel of the playback buffer should play to
	 * @param lChannelAmplitudes : amplitudes corresponding to the lChannelOutputChannels param.
	 * @param rChannelOutputChannels : an array of output channels which the right channel of the playback buffer should play to
	 * @param rChannelAmplitudes
	 * @param playbackRate : amplitudes corresponding to the rChannelOutputChannels param.
	 * @return a new SoundNode
	 */

	public SoundNode createStereoSoundNode(int bufferNumber, boolean doLoop,
			int[] lChannelOutputChannels, float[] lChannelAmplitudes, int[] rChannelOutputChannels, float[] rChannelAmplitudes, float playbackRate) {
		
		//sanity check the buffer that's been requested.
		if (!_bufferMap.containsKey(bufferNumber)) return null;
		
		float[][] amps = new float[2][];
		amps[0] = lChannelAmplitudes;
		amps[1] = rChannelAmplitudes;

		int[][] outChannels = new int[2][];
		outChannels[0] = lChannelOutputChannels;
		outChannels[1] = rChannelOutputChannels;

		// instantiate a new SoundNode and remember it by its group ID.
		// when that group ID is freed on the server we'll know this SoundNode
		// is dead.
		SoundNode sn;
		synchronized (this) {
			sn = new SoundNode(this, bufferNumber, 2, doLoop,
					outChannels, amps, playbackRate);
			_soundNodes.put(sn.getGroup(), sn);
		}

		return sn;
	}

	
	
	
	//*********************************
	// debug output helpers
	//*********************************

	boolean _doDebug = true;

	public void showDebugOutput(boolean state) {
		_doDebug = state;
	}

	public void debugPrint(String s) {
		if (_doDebug)
			logger.debug(s);
	}

	public void debugPrintln(String s) {
		if (_doDebug)
			logger.debug(s);
	}

	
	
	//*********************************
	// Event Handling Methods
	//*********************************
	
	protected void handleServerBooted() {
		logger.info("SCSC: scsynth is booted.");
		//reinit data.
		//this.init();
		//notify client.
		if (_notifyListener != null) { _notifyListener.receiveNotification_ServerRunning(); }
		if (_controlPanel != null && _controlPanel._statsDisplay != null) {
				_controlPanel._statsDisplay.receiveNotification_ServerRunning();
		}
	}
	
	protected void handleServerStatusUpdate(int numUgens, int numSynths, int numGroups, int numSynthdefs, float avgCPU, float peakCPU) {
		_prevPingResponseTime = new Date();
		
		//if (!_serverLive || !_serverBooted) {
		if (!_serverLive) {
			if(debugging){
				logger.info("SCSC: scsynth is live.");
			}
			_serverLive = true;
			this.init();
		}
		
		if (_notifyListener != null) { _notifyListener.receiveNotification_ServerStatus(avgCPU, peakCPU, numSynths); }
		if (_controlPanel != null && _controlPanel._statsDisplay != null) {
			_controlPanel._statsDisplay.receiveNotification_ServerStatus(avgCPU, peakCPU, numSynths);
			_controlPanel._statsDisplay.notify_currentPolyphony(_soundNodes.size() / (float)_maxPolyphony);
		}
		
		//debugPrintln("status latency: " + (_prevPingResponseTime.getTime() - _prevPingRequestTime.getTime()));
	}
	
	
	//SCSoundControl starts up a thread to make sure the server is running.
	public void run() {
		Date curTime;
		while(true) {
			curTime = new Date();
			
			//System.out.println("currentTime: "+ curTime.getTime() +" prevTime: "+ _prevPingRequestTime.getTime() +" diff: " + (curTime.getTime() - _prevPingRequestTime.getTime()) +" interval: "+ _scsynthPingInterval);

			//if the previous status request is still pending...
			if (_serverLive && (_prevPingRequestTime.getTime() > _prevPingResponseTime.getTime())) {
				//We've not yet heard back from the previous status request.
				//Have we timed out?
				if (curTime.getTime() - _prevPingRequestTime.getTime() > _serverResponseTimeout) {
					//We've timed out on the previous status request.
					logger.info("SCSC: Timed out on previous status request.");

					if (_notifyListener != null) {
						_notifyListener.receiveNotification_ServerStopped();
					}
					if (_controlPanel != null && _controlPanel._statsDisplay != null) {
						_controlPanel._statsDisplay.receiveNotification_ServerStopped();
					}

					_serverLive = false;
					_serverBooted = false;
				}
				//else we just keep waiting for a response or a timeout
			}
			//the previous status request is NOT still pending. Is it time to send another?
			else if (curTime.getTime() - _prevPingRequestTime.getTime() > _scsynthPingInterval) {
				//It's time to send another status request.
				
				//generally, ping with a /status message.
				//but, if we're live but not booted, query node 1 (the sign that init completed)
				if (_serverLive && !_serverBooted) { 
					sendMessage("/notify", new Object[] { 1 });
					sendMessage("/n_query", new Object[]{_motherGroupID});
					//System.out.println("Querying SCSC mother node");
					//debugPrintln("Querying SCSC mother node");
				}
				else {
					//if the server's booted, request a status update.
					sendMessage("/status"); 
				}
				
				_prevPingRequestTime = new Date();
			}
			//it's not time to send, and we're not watching 
			//for a reply, so go to sleep until it's time to ping again
			else {
				long sleeptime = Math.max(_scsynthPingInterval - (curTime.getTime() - _prevPingRequestTime.getTime()), 0);
				//System.out.println("sleep " + sleeptime);
				try {
					Thread.sleep(sleeptime);
				} catch (InterruptedException e) {
					//NOTE this thread shouldn't get interrupted.
					e.printStackTrace();
				}			
			}
			
			
		}
	}

	
	//*********************************	
	//modify server ping settings: 
	//*********************************
	
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
	

}


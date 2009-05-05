package net.electroland.broadcast.server;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * This class is a socket server that collects connections from Flash clients,
 * and forwards all messages it has been asked to deliver to ALL of the clients 
 * that have connected to it.
 * 
 * The architecture looks like this:
 * 
 *         MessageCreator (presumably Eitan's code)
 *                |
 *    XMLSocketBroadcaster
 *    |     |     |     |     |
 * Flash Flash Flash Flash Flash
 * 
 * For the most part, communication is unidirectional.  Each Flash client makes
 * an XMLSocket request like this:
 * 
 * var xmlsock:XMLSocket = new XMLSocket();
 * xmlsock.connect(XMLSocketBroadcasterIPAddress, port);
 * 
 * The this class accepts the connection.  From thereon out, if send(XMLMessage)
 * is called, any Flash client that is connected will receive the message.
 * 
 * @author geilfuss
 *
 */

public class XMLSocketBroadcaster implements Runnable {

	// all the folks we'll be broadcasting too.
	private CopyOnWriteArrayList<Socket> listeners;

	boolean isRunning = true;
	// Anyone who wants to receive broadcasts, needs to register as a listener.
	// 'listenerRegistrationThread' actively listens to port 'port' for 
	// lisener registrations.  Registration is simply a persistent socket
	// connection.
	private Thread listenerRegistrationThread;
	private int port;

	SenderThread senderThread = new SenderThread();
	
	/**
	 * @param port - the port this broadcaster will listen on for clients.
	 */
	public XMLSocketBroadcaster(int port){
		System.out.println("XMLSocketBroadcaster(): Starting XMLSocketBroadcaster on port " + 
							port + " at " + new Date());
		this.listeners = new CopyOnWriteArrayList<Socket>();
		this.port = port;
	}

	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run() {
		try{
			ServerSocket registrar = new ServerSocket(port);
			while (listenerRegistrationThread != null){
				Socket listener = registrar.accept();
				System.out.println("XMLSocketBroadcaster.run(): accepted connection from: " + 
									listener.getInetAddress());
				listener.setTcpNoDelay(true);
				listeners.add(listener);
			}
		}catch(IOException e){
			System.out.println("XMLSocketBroadcaster.run(): ERROR:" + e);
			e.printStackTrace(System.out);
			// restart !?!?
		}
	}

	/**
	 * 
	 * @param message
	 */
	public void send(XMLSocketMessage message){
		senderThread.send(message.toXML());
	}
	
	public void sendToListeners(String xml){
//		String xml = message.toXML();
		// listeners is a Vector, so shouldn't need to sync
		for (Socket listener : listeners){
			// if performance becomes an issue, this should be multithreaded.
			try{
				PrintWriter pw = new PrintWriter(listener.getOutputStream());
				pw.write(FlexBroadcasterUtil.XML_HEADER);
				pw.write(xml);
				pw.write((byte)0); 	// this is the message terminus that Flash's
									// XMLSocket expects.
				
				//ds change per brad
				if (pw.checkError()){
					System.out.println("XMLSocketBroadcaster.send(): paring listener.");
					listeners.remove(listener);
				}
				
				//ds change per brad
				pw.flush();
				
			}catch(IOException e){
				System.out.println("XMLSocketBroadcaster.send(): ERROR:" + e);
				e.printStackTrace();
			}
		}
	}
	
	public class SenderThread extends Thread {
		public Object lock = new Object();
		public String xml;
		
		public void run() {
			while(isRunning) {
				synchronized(lock) {
					try {
						lock.wait();
					} catch (InterruptedException e) {
					}
					sendToListeners(xml);
				}				
			}
		}
		public void send(String xml) { 
			synchronized(lock) {
				this.xml = xml;
				lock.notify();
			}
			
		}
	}

	/**
	 * 
	 *
	 */
	public synchronized void start() {
		senderThread.start();
		listenerRegistrationThread = new Thread(this);
		listenerRegistrationThread.start();
	}

	/**
	 * 
	 *
	 */
	public synchronized void stop(){
		// kill the thread.
		isRunning = false;
		senderThread.notify();
		listenerRegistrationThread.interrupt();
		listenerRegistrationThread = null;

		// close all existing socket connections
		for (Socket listener : listeners){
			try {
				listener.close();
			} catch (IOException e) {
				System.out.println("XMLSocketBroadcaster.stop(): ERROR:" + e);
				e.printStackTrace();
			}
		}
	}
}
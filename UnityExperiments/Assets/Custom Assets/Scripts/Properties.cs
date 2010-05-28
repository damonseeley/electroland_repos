using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class Properties {
	
	private Dictionary<string, string> dictionary = new Dictionary<string, string>();
	
	public Properties(string filename){
		foreach(string line in File.ReadAllLines(filename)){	// for every line in file...
			if(!string.IsNullOrEmpty(line)){					// ignore empty lines
				if(!line.StartsWith("#")){						// ignore comments
					if(line.Contains("=")){						// must have assignment operator
						
						int index = line.IndexOf("=");
						string key = line.Substring(0, index).Trim();
						string value = line.Substring(index + 1).Trim();
						if ((value.StartsWith("\"") && value.EndsWith("\"")) || (value.StartsWith("'") && value.EndsWith("'"))){
							value = value.Substring(1, value.Length - 2);
						}
						dictionary.Add(key, value);

					}
				}
			}
		}
	}
	
	public bool contains(string key){
		return dictionary.ContainsKey(key);
	}
	
	public string getProperty(string key){
		if(dictionary.ContainsKey(key)){
			return dictionary[key];
		}
		return null;
	}
	
}

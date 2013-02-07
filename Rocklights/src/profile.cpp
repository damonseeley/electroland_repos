//
// Profile
// Provides access to INI profile values
//

#include "profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <ctype.h>
#include <string.h>



struct SProfileEntry
{
	unsigned int hash;
	const char* name;
	const char* value;
	SProfileEntry* next;
};


CProfile *CProfile::theProfile = NULL;
CProfile::CProfile()
{
	m_strings = NULL;
	m_offset = 0;
	memset(m_hash, 0x00, sizeof(m_hash));
  theProfile = this;
}


CProfile::~CProfile()
{
	while(m_strings)
	{
		char* strings = m_strings;
		m_strings = *(char**) m_strings;
		delete [] strings;
	}

	for(unsigned int i = 0; i < 1021; i++)
	{
		while(m_hash[i])
		{
			SProfileEntry* entry = m_hash[i];
			m_hash[i] = entry->next;
			delete entry;
		}
	}
}


bool CProfile::Load(const char* filename)
{
	bool ret = false;
	FILE* file = NULL;
	char* data = NULL;
	unsigned int size = 0;

	// Open file
	if(!(file = fopen(filename, "rb")))
		goto LFail;

	// Get file size
	fseek(file, 0, SEEK_END);
	size = (unsigned int) ftell(file);

	// Read file
	if(!(data = new char[size + 1]))
		goto LFail;

	fseek(file, 0, SEEK_SET);
	fread(data, 1, size, file);

	data[size] = 0;


	// Parse file
	m_pch = data;

	while(*m_pch)
	{
		const char *name = GetToken();

		if(!name)
			break;

		SkipWhitespace();

		if('=' != m_pch[0])
			goto LFail;

		m_pch++;

		const char *value = GetString();

		if(!value)
			goto LFail;

		SProfileEntry* entry = new SProfileEntry;

		if(!entry)
			goto LFail;

		unsigned int hash = Hash(name);
		unsigned int index = hash % 1021;

		entry->hash = hash;
		entry->name = name;
		entry->value = value;
		entry->next = m_hash[index];

		m_hash[index] = entry;
	}

	ret = true;
	goto LDone;

LFail:
	ret = false;
	goto LDone;

LDone:
	if(file)
		fclose(file);

	delete [] data;
	return ret;
}


const char* CProfile::String(const char* name, const char* def)
{
	SProfileEntry* entry = Find(name);
	return entry ? entry->value : def;
}


double CProfile::Double(const char* name, double def)
{
	SProfileEntry* entry = Find(name);
	return entry ? atof(entry->value) : def;
}


float CProfile::Float(const char* name, float def)
{
	SProfileEntry* entry = Find(name);
	return entry ? (float) atof(entry->value) : def;
}


int CProfile::Int(const char* name, int def)
{
	SProfileEntry* entry = Find(name);
	return entry ? atoi(entry->value) : def;
}

bool CProfile::Bool(const char* name, bool def)
{
	SProfileEntry* entry = Find(name);
	return entry ? (strcmp(entry->value, "false") != 0) : def;
}

void CProfile::SkipWhitespace()
{
	while(*m_pch)
	{
		if(m_pch[0] == ' ' || m_pch[0] == '\t' || m_pch[0] == '\n' || m_pch[0] == '\r')
		{
			// Whitespace
			m_pch++;
		}
		else if(m_pch[0] == '#')
		{
			// insilio txt style comment
			m_pch += 1;

			while(*m_pch)
			{
				if(m_pch[0] == '\n' || m_pch[0] == '\r')
					break;

				m_pch++;
			}
		}
		else if(m_pch[0] == '/' && m_pch[1] == '/')
		{
			// C++ style comment
			m_pch += 2;

			while(*m_pch)
			{
				if(m_pch[0] == '\n' || m_pch[0] == '\r')
					break;

				m_pch++;
			}
		}
		else if(m_pch[0] == '/' && m_pch[1] == '*')
		{
			// C style comment
			m_pch += 2;

			while(*m_pch)
			{
				if(m_pch[0] == '*' && m_pch[1] == '/')
					break;

				m_pch++;
			}
		}
		else
		{
			break;
		}
	}
}


const char* CProfile::GetToken()
{
	SkipWhitespace();

	if(!isalpha(m_pch[0]) && ('_' != m_pch[0]))
		return NULL;

	const char* token = m_pch;

	while(*m_pch)
	{
		if(!isalnum(m_pch[0]) && ('_' != m_pch[0]))
			break;

		m_pch++;
	}

	return AddString(token, (unsigned int) (m_pch - token));
}


const char* CProfile::GetString()
{
	SkipWhitespace();
	const char* str = m_pch;

	if('"' == m_pch[0])
	{
		m_pch++;

		while(*m_pch)
		{
			if('"' == m_pch[0])
			{
				m_pch++;
				break;
			}

			m_pch++;
		}

		return AddString(str + 1, (unsigned int) (m_pch - str - 2));
	}
	else
	{
		while(*m_pch)
		{
			if(' ' == m_pch[0] || '\t' == m_pch[0] || '\n' == m_pch[0] || '\r' == m_pch[0])
				break;
			else if('#' == m_pch[0])
				break;
			else if('/' == m_pch[0] && '/' == m_pch[1])
				break;
			else if('/' == m_pch[0] && '*' == m_pch[1])
				break;

			m_pch++;
		}

		return AddString(str, (unsigned int) (m_pch - str));
	}
}


const char* CProfile::AddString(const char* str, unsigned int len)
{
	if(!m_strings || (len + 1 + m_offset > 4096))
	{
		char* strings = new char[4096];
		*(char**) strings = m_strings;
		m_strings = strings;
		m_offset = sizeof(char*);
	}

	memcpy(m_strings + m_offset, str, len);
	char* add = m_strings + m_offset;
	add[len] = 0;
	m_offset += len + 1;
	return add;
}


unsigned int CProfile::Hash(const char* name)
{
	unsigned int hash = 0;

	for(const char* pch = name; *pch; pch++)
		hash = hash * 0xFFF1 + tolower(*pch);

	return hash;
}


SProfileEntry* CProfile::Find(const char* name)
{
	unsigned int hash = Hash(name);

	for(SProfileEntry* entry = m_hash[hash % 1021]; entry; entry = entry->next)
	{

//		if((entry->hash == hash) && !strcasecmp(entry->name, name))
		if((entry->hash == hash) && !stricmp(entry->name, name))
			return entry;
	}

	return NULL;
}

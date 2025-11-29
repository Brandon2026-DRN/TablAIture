"""
Formatting utilities for tablature output.
"""

from typing import List, Dict, Any
from ..inference.tablature_inference import NoteEvent


def notes_to_json(notes: List[NoteEvent]) -> List[Dict[str, Any]]:
    """
    Convert NoteEvent list to JSON-serializable format.
    
    Args:
        notes: List of NoteEvent objects
        
    Returns:
        List of dictionaries with required format
    """
    return [
        {
            "time": float(note.time),
            "string": int(note.string),
            "fret": int(note.fret),
            "pitch_hz": float(note.pitch_hz),
            "confidence": float(note.confidence)
        }
        for note in notes
    ]


def format_plain_text_tab(notes: List[NoteEvent], n_strings: int = 6) -> str:
    """
    Format notes as plain-text tablature.
    
    Args:
        notes: List of NoteEvent objects
        n_strings: Number of strings (default: 6)
        
    Returns:
        Plain-text tablature string
    """
    if not notes:
        return ""
    
    # Group notes by time
    time_groups: Dict[float, List[NoteEvent]] = {}
    for note in notes:
        if note.time not in time_groups:
            time_groups[note.time] = []
        time_groups[note.time].append(note)
    
    # Sort by time
    sorted_times = sorted(time_groups.keys())
    
    # Build tab lines
    tab_lines = []
    string_names = ["E", "A", "D", "G", "B", "E"]
    
    for time in sorted_times:
        time_notes = time_groups[time]
        
        # Create tab for this time
        tab_row = [""] * n_strings
        for note in time_notes:
            string_idx = note.string - 1  # Convert to 0-indexed
            if 0 <= string_idx < n_strings:
                tab_row[string_idx] = str(note.fret)
        
        # Format as tab lines
        for i, (string_name, fret) in enumerate(zip(string_names, tab_row)):
            if fret:
                tab_lines.append(f"{string_name}|-{fret}-")
            else:
                tab_lines.append(f"{string_name}|---")
        
        tab_lines.append("")  # Blank line between time steps
    
    return "\n".join(tab_lines)


def format_musicxml(notes: List[NoteEvent]) -> str:
    """
    Format notes as MusicXML (simplified).
    
    Args:
        notes: List of NoteEvent objects
        
    Returns:
        MusicXML string
    """
    # Simplified MusicXML structure
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC',
        '    "-//Recordare//DTD MusicXML 3.1 Partwise//EN"',
        '    "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="3.1">',
        '  <part-list>',
        '    <score-part id="P1">',
        '      <part-name>Guitar</part-name>',
        '    </score-part>',
        '  </part-list>',
        '  <part id="P1">'
    ]
    
    # Group notes by measure (simplified: 4/4 time, 1 measure = 4 beats)
    # This is a simplified version - full MusicXML would need proper measure/beat structure
    current_measure = 1
    xml_lines.append(f'    <measure number="{current_measure}">')
    xml_lines.append('      <attributes>')
    xml_lines.append('        <divisions>480</divisions>')
    xml_lines.append('        <time>')
    xml_lines.append('          <beats>4</beats>')
    xml_lines.append('          <beat-type>4</beat-type>')
    xml_lines.append('        </time>')
    xml_lines.append('        <clef>')
    xml_lines.append('          <sign>TAB</sign>')
    xml_lines.append('        </clef>')
    xml_lines.append('      </attributes>')
    
    for note in notes:
        # Convert time to divisions (480 divisions = 1 quarter note)
        divisions = int(note.time * 480)  # Assuming 1 second = 1 quarter note
        
        xml_lines.append('      <note>')
        xml_lines.append(f'        <string>{note.string}</string>')
        xml_lines.append(f'        <fret>{note.fret}</fret>')
        xml_lines.append(f'        <duration>{divisions}</duration>')
        xml_lines.append('        <type>quarter</type>')
        xml_lines.append('      </note>')
    
    xml_lines.append('    </measure>')
    xml_lines.append('  </part>')
    xml_lines.append('</score-partwise>')
    
    return "\n".join(xml_lines)


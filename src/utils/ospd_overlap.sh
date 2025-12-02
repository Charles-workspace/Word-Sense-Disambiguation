for f in ap*; do
    if grep -iqwE "sound|concert|dance|singer" "$f"; then
        if grep -iqwE "rain|fire|park|land" "$f"; then
            echo "=================================================================="
            echo "FILE: $f  →  BOTH SENSES PRESENT (musicforest)"
            echo "------------------------------------------------------------------"
            echo "[MUSIC sense cues]"
            grep -iwn --color=always "sound\|concert\|dance\|singer" "$f" | head -8
            echo "------------------------------------------------------------------"
            echo "[FOREST sense cues]"
            grep -iwn --color=always "rain\|fire\|park\|land" "$f" | head -8
            echo -e "\n\n"
        fi
    fi
done | less -R
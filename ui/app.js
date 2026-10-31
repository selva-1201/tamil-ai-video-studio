(function () {
      // -----------------------------------------------------------------------
      // Application State
      // -----------------------------------------------------------------------
      var state = {
        activeScreen: 'create',
        duration: 'auto',
        mode: 'cinematic',
        aspect: '9:16',
        voiceProfileId: 'default_tamil_narrator',
        mood: 'emotional',
        engine: 'wan22',
        preset: 'balanced',
        resolution: '480p',
        jobCancelled: false,
        simTimer: null,
        cinemaTimer: null,
        audioCtx: null,
        currentJob: {
          id: 'job_94f2a81c',
          title: 'மழையில் நனையாத கனவு (Unbroken Dream in the Rain)',
          target_seconds: 45,
          final_duration_sec: 43.5,
          mood: 'emotional',
          music_prompt: 'Emotional South Indian cinematic score, solo bamboo flute (pullanguzhal) and warm cello, gentle acoustic guitar arpeggios, slow 72 BPM tempo, poignant and hopeful mood, instrumental, no vocals',
          character_description_en: 'A 9-year-old Tamil schoolboy named Karthik with warm brown skin, expressive dark eyes, drenched black hair, wearing a faded navy-blue school uniform shirt, khaki shorts, rubber sandals, and carrying a worn black canvas schoolbag clutched tightly to his chest',
          scenes: [
            {
              id: 1,
              narration_ta: 'காலை நேரத்துல வானம் இருண்டு, விடாம கனமழை பெய்யத் தொடங்குச்சு. ஆனாலும் அந்த சின்ன பையன், தன் புத்தகப் பையை மார்போடு இறுக்கிப் பிடிச்சுக்கிட்டு நடந்தான்.',
              visual_prompt_en: 'A 9-year-old Tamil schoolboy named Karthik in a faded navy-blue school uniform shirt and khaki shorts stepping onto a muddy village road in Tamil Nadu during a heavy morning monsoon downpour. Low-angle tracking camera following his sandals splashing through red-earth puddles, 35mm lens. Ambient sound: heavy rain pouring on palm leaves, splashing water puddles, distant thunder.',
              est_seconds: 9,
              cer: 0.014,
              qa_passed: true
            },
            {
              id: 2,
              narration_ta: 'குடை கூட இல்லாம, மழைத் துளிகள் முகத்துல விழ, அவன் கண்கள்ல ஒரு பெரிய கனவு தெரிஞ்சது. படிப்பு மட்டும்தான் தன் குடும்பத்தைக் காப்பாத்தும்னு அவன் உறுதியா நம்பினான்.',
              visual_prompt_en: 'Close-up dolly shot of Karthik looking ahead with quiet determination as raindrops fall off his eyelashes. Blurred green paddy fields and swaying coconut trees in the monsoon wind behind him, shallow depth of field, 50mm prime lens. Ambient sound: steady rain droplets hitting fabric, monsoon breeze, faint village temple bell.',
              est_seconds: 9,
              cer: 0.019,
              qa_passed: true
            },
            {
              id: 3,
              narration_ta: 'வழியில மரத்தடியில ஒதுங்க நினைச்சாலும், பள்ளிக்கூட மணி சத்தம் அவனை வேகமா ஓட வச்சது. புத்தகங்கள் நனையாம இருக்க, தன் சட்டையால பையை மூடிக்கிட்டான்.',
              visual_prompt_en: 'Karthik shielding his canvas schoolbag beneath his uniform shirt as he jogs past a roadside banyan tree. Dynamic side-tracking medium shot, glistening wet asphalt reflecting the overcast sky, 35mm lens. Ambient sound: rapid footsteps splashing through water, rustling banyan leaves, distant school bell ringing.',
              est_seconds: 8,
              cer: 0.021,
              qa_passed: true
            },
            {
              id: 4,
              narration_ta: 'கடைசியா பள்ளிக்கூட வாசலை அடைஞ்சப்போ, ஆசிரியர் அவனைப் பார்த்து பெருமையா புன்னகைச்சார். அந்த ஒரு சிரிப்புல, அவன் நனைஞ்ச குளிரும் கஷ்டமும் உடனே மறைஞ்சு போச்சு.',
              visual_prompt_en: 'Karthik arriving under the sheltered veranda of a traditional Tamil Nadu government school with yellow pillars. A kind Tamil teacher in a crisp white shirt places a warm hand on the boy shoulder and hands him a dry towel, 50mm lens. Ambient sound: rain dripping from clay roof tiles, gentle classroom murmur.',
              est_seconds: 9,
              cer: 0.016,
              qa_passed: true
            },
            {
              id: 5,
              narration_ta: 'வகுப்பறையில அமர்ந்து நோட்டுப் புத்தகத்தைத் திறந்தப்போ, அவன் முகத்துல நம்பிக்கை வெளிச்சம் பரவுச்சு. உழைப்பும் கல்வியும் என்னைக்குமே தோற்காது.',
              visual_prompt_en: 'Karthik sitting at a wooden classroom bench beside a rain-streaked window, opening a dry notebook with a hopeful smile as golden sunlight breaks through the clouds outside. Gentle crane-up shot, volumetric sunbeams, 50mm lens. Ambient sound: soft pencil scribbling on paper, birds chirping after rain.',
              est_seconds: 8,
              cer: 0.018,
              qa_passed: true
            }
          ]
        },
        stages: [
          { key: 'plan', name: 'Stage 1 · Planner LLM (Qwen3-8B-Instruct GGUF Q4_K_M)', detail: 'Tanglish → validated Tamil script JSON · 2.4s · GPU0 Peak 5.2 GB', status: 'done' },
          { key: 'voice', name: 'Stage 2 · Tamil TTS (AI4Bharat IndicF5 24 kHz) + Whisper QA', detail: '10/10 sentences QA passed (CER 1.8%) · −16 LUFS · 43.5s · GPU0 Peak 3.8 GB', status: 'done' },
          { key: 'music', name: 'Stage 4 · Instrumental Score (ACE-Step 1.5 Turbo · shift=3.0)', detail: '45.5s bamboo flute & cello score · 8 ODE steps · GPU0 Peak 4.9 GB', status: 'done' },
          { key: 'video', name: 'Stage 5 · Video Engine (Wan 2.2 A14B Q8 Split + MMAudio)', detail: 'Shot 6/9 rendering (480p · last-frame I2V continuity · keyframe reset every 3 clips)', status: 'running' },
          { key: 'mix', name: 'Stage 6 · FFmpeg Mastering & Sidechain Ducking', detail: 'Voice 0 dB · Music −18 dB (ducked) · Ambience −24 dB · H.264 yuv420p +faststart', status: 'pending' },
          { key: 'drive', name: 'Stage 7 · Google Drive Sync (rclone OAuth gdrive:)', detail: 'Background upload to gdrive:TamilVideoStudio/2026-10-31/job_94f2a81c/', status: 'pending' }
        ],
        logs: [
          '[09:48:12] Applied T4 compatibility patches: flash_attn -> PyTorch SDPA shim, bfloat16 -> float16 autocast.',
          '[09:48:14] Stage 1 START: Loaded Qwen3-8B-Q4_K_M.gguf on cuda:0 (5.18 GB VRAM).',
          '[09:48:17] Stage 1 COMPLETE: Validated 5 scenes (96 Tamil script words, 0 Latin chars). Unloaded LLM -> VRAM 0.0 GB.',
          '[09:48:18] Stage 2 START: Loaded ai4bharat/IndicF5 (24 kHz) + faster-whisper large-v3-turbo (fp16).',
          '[09:48:31] Stage 2 QA: All 10 sentences passed CER threshold (<= 0.18). Normalized to -16.0 LUFS (43.5s). Unloaded TTS.',
          '[09:49:02] Stage 3 Review Gate approved by user. Starting Stage 4 (ACE-Step 1.5 Turbo, shift=3.0, steps=8)...',
          '[09:49:10] Stage 4 COMPLETE: Generated 45.5s instrumental track (music.wav). Unloaded ACE-Step -> VRAM 0.0 GB.',
          '[09:49:12] Stage 5 START: Wan 2.2 A14B GGUF split (Transformer -> cuda:0, UMT5-XXL + VAE -> cuda:1).',
          '[09:50:45] Stage 5 Checkpoint: Shot 5/9 + MMAudio ambient track saved and synced to gdrive:TamilVideoStudio.'
        ],
        library: [
          {
            id: 'job_94f2a81c',
            title: 'மழையில் நனையாத கனவு (Unbroken Dream in the Rain)',
            subtitle: 'காலை நேரத்துல வானம் இருண்டு, விடாம கனமழை பெய்யத் தொடங்குச்சு...',
            duration: '43.5 s',
            engine: 'Wan 2.2 A14B Q8 + MMAudio',
            aspect: '9:16',
            date: '2026-10-31',
            driveUri: 'gdrive:TamilVideoStudio/2026-10-31/job_94f2a81c/'
          },
          {
            id: 'job_71c9e30b',
            title: 'கடற்கரை மாலைப் பொழுது (Marina Beach Family Evening)',
            subtitle: 'அலைகள் கரையைத் தொடும் மாலை நேரத்துல, குடும்பத்தோடு சிரிச்சுப் பேசுற நிமிஷம்...',
            duration: '31.2 s',
            engine: 'LTX-2.3 22B Distilled GGUF',
            aspect: '16:9',
            date: '2026-10-31',
            driveUri: 'gdrive:TamilVideoStudio/2026-10-31/job_71c9e30b/'
          },
          {
            id: 'job_38a4d11f',
            title: 'தஞ்சாவூர் பெரிய கோவில் வரலாறு (Brihadeeswarar Chola Architecture)',
            subtitle: 'ஆயிரம் வருஷம் கடந்தும் நிமிர்ந்து நிக்குற தஞ்சை பெரிய கோவிலின் ரகசியம்...',
            duration: '58.4 s',
            engine: 'Wan 2.2 A14B + LightX2V 4-Step',
            aspect: '9:16',
            date: '2026-10-30',
            driveUri: 'gdrive:TamilVideoStudio/2026-10-30/job_38a4d11f/'
          }
        ],
        voices: [
          {
            id: 'default_tamil_narrator',
            name: 'Default Tamil Studio Voice (24 kHz)',
            duration: '8.5 s',
            transcript_ta: 'வணக்கம், இது தமிழ் ஏஐ வீடியோ ஸ்டுடியோவின் இயற்கையான குரல் பதிவு.',
            verified: true
          },
          {
            id: 'user_cloned_karthik',
            name: 'Karthik — Warm Storyteller (Cloned)',
            duration: '10.4 s',
            transcript_ta: 'காலை நேரத்துல அமைதியாப் பெய்யுற மழை மனசுக்கு ஒரு புது உற்சாகத்தைக் கொடுக்குது.',
            verified: true
          },
          {
            id: 'user_cloned_ananya',
            name: 'Ananya — Expressive Narrator (Cloned)',
            duration: '9.8 s',
            transcript_ta: 'நம்ம கனவுகள் நனவாக தினமும் சின்ன சின்ன முயற்சிகளைத் தொடர்ந்து செய்யணும்.',
            verified: true
          }
        ]
      };

      // -----------------------------------------------------------------------
      // Utility: In-Page Monochrome Toast (No alert/confirm/prompt)
      // -----------------------------------------------------------------------
      function showToast(msg) {
        var container = document.getElementById('toastContainer');
        if (!container) return;
        var el = document.createElement('div');
        el.className = 'toast';
        el.textContent = msg;
        container.appendChild(el);
        setTimeout(function () {
          if (el.parentNode) el.parentNode.removeChild(el);
        }, 3800);
      }

      // -----------------------------------------------------------------------
      // Navigation Tabs
      // -----------------------------------------------------------------------
      function switchScreen(screenId) {
        state.activeScreen = screenId;
        var tabs = document.querySelectorAll('.nav-tab');
        for (var i = 0; i < tabs.length; i++) {
          var isMatch = tabs[i].getAttribute('data-screen') === screenId;
          tabs[i].classList.toggle('active', isMatch);
          tabs[i].setAttribute('aria-selected', isMatch ? 'true' : 'false');
        }
        var screens = document.querySelectorAll('.screen');
        for (var j = 0; j < screens.length; j++) {
          screens[j].classList.toggle('active', screens[j].id === 'screen-' + screenId);
        }
        if (screenId === 'review') renderReviewScenes();
        if (screenId === 'progress') renderStepperAndLogs();
        if (screenId === 'library') {
          renderLibraryGrid();
          drawCinemaFrame(0);
        }
        if (screenId === 'voices') {
          renderSavedVoices();
          drawWaveformCanvas();
        }
      }

      var navButtons = document.querySelectorAll('.nav-tab');
      for (var i = 0; i < navButtons.length; i++) {
        navButtons[i].addEventListener('click', function (e) {
          var target = e.currentTarget.getAttribute('data-screen');
          if (target) switchScreen(target);
        });
      }

      // -----------------------------------------------------------------------
      // Screen 1: Create Controls & Chip Selectors
      // -----------------------------------------------------------------------
      function bindChipGroup(containerId, stateKey, onChange) {
        var container = document.getElementById(containerId);
        if (!container) return;
        var chips = container.querySelectorAll('.chip');
        for (var i = 0; i < chips.length; i++) {
          chips[i].addEventListener('click', function (e) {
            var val = e.currentTarget.getAttribute('data-val');
            state[stateKey] = val;
            for (var k = 0; k < chips.length; k++) {
              chips[k].classList.toggle('active', chips[k] === e.currentTarget);
            }
            if (onChange) onChange(val);
          });
        }
      }

      bindChipGroup('chipGroupDuration', 'duration');
      bindChipGroup('chipGroupMode', 'mode', function (val) {
        var uploadBox = document.getElementById('modeBUploadBox');
        var previewBtn = document.getElementById('btnPreview5sTalking');
        if (uploadBox) uploadBox.style.display = val === 'talking' ? 'block' : 'none';
        if (previewBtn) previewBtn.style.display = val === 'talking' ? 'inline-flex' : 'none';
      });
      bindChipGroup('chipGroupAspect', 'aspect');

      document.getElementById('btnLoadSampleRain').addEventListener('click', function () {
        document.getElementById('promptInput').value = 'oru chinna paiyan mazhai la school ku nadandhu poguraan, emotional ah';
        document.getElementById('selectMood').value = 'emotional';
        showToast('Loaded acceptance test Tanglish prompt.');
      });

      document.getElementById('btnLoadSampleBeach').addEventListener('click', function () {
        document.getElementById('promptInput').value = 'oru kudumbam marina beach la evening sundal saapttu santhoshama pesitu irukaanga, cinematic warm light';
        document.getElementById('selectMood').value = 'cinematic';
        showToast('Loaded Tanglish Marina Beach prompt.');
      });

      document.getElementById('btnGeneratePlan').addEventListener('click', function () {
        var rawPrompt = document.getElementById('promptInput').value.trim();
        if (!rawPrompt) {
          showToast('Please enter a Tanglish, Tamil, or English prompt first.');
          return;
        }
        state.engine = document.getElementById('selectVideoEngine').value;
        state.voiceProfileId = document.getElementById('selectVoiceProfile').value;
        state.mood = document.getElementById('selectMood').value;

        var pLow = rawPrompt.toLowerCase();
        if (pLow.indexOf('kudumbam') !== -1 || pLow.indexOf('beach') !== -1) {
          state.currentJob.title = 'கடற்கரை மாலைப் பொழுது (Family Evening by the Sea)';
          state.currentJob.music_prompt = 'Warm South Indian acoustic guitar and flute melody with gentle ocean wave rhythm, 80 BPM, joyful family atmosphere, instrumental, no vocals';
          state.currentJob.character_description_en = 'A warm Tamil family of four—father in a linen shirt, mother in a pastel cotton saree, and two cheerful children—at Marina Beach during golden hour';
          state.currentJob.scenes[0].narration_ta = 'மாலை நேரத்து சூரியன் கடலுக்குள்ள மெல்ல மறையத் தொடங்குச்சு. கடற்கரைக் காத்துல அந்தக் குடும்பத்தோட சிரிப்பு சத்தம் நிறைஞ்சிருந்தது.';
          state.currentJob.scenes[0].visual_prompt_en = state.currentJob.character_description_en + ', walking barefoot along the wet shoreline as golden sunset reflections shimmer on the waves, 35mm lens. Ambient sound: gentle ocean waves breaking, seagulls, warm laughter.';
        }

        var etaSec = state.engine === 'ltx23' ? 210 : (state.engine === 'wan22_lora' ? 310 : 435);
        document.getElementById('reviewEtaText').textContent =
          '~' + Math.floor(etaSec / 60) + 'm ' + (etaSec % 60) + 's (9 shots · ' + state.engine.toUpperCase() + ' · 2×T4 Split)';

        showToast('Stage 1 (Qwen3 GGUF) & Stage 2 (IndicF5 + Whisper QA) completed! Review script & voice.');
        switchScreen('review');
      });

      // -----------------------------------------------------------------------
      // Screen 2: Review Gate Rendering & Web Audio Voice Preview
      // -----------------------------------------------------------------------
      function renderReviewScenes() {
        var container = document.getElementById('reviewScenesContainer');
        if (!container) return;
        container.innerHTML = '';

        var scenes = state.currentJob.scenes;
        for (var i = 0; i < scenes.length; i++) {
          (function (scene, idx) {
            var card = document.createElement('div');
            card.className = 'glass-subpanel scene-card';

            var badgeClass = scene.qa_passed ? 'badge-solid' : 'badge-dashed';
            var badgeText = scene.qa_passed
              ? ('Whisper QA PASS · CER ' + (scene.cer * 100).toFixed(1) + '%')
              : ('Whisper QA RETRY · CER ' + (scene.cer * 100).toFixed(1) + '%');

            card.innerHTML =
              '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:10px;">' +
                '<div style="font-weight:600; font-size:14px;">Scene ' + scene.id + ' · Est. ' + scene.est_seconds + ' s (2 Shots)</div>' +
                '<div style="display:flex; gap:8px; align-items:center;">' +
                  '<span class="badge ' + badgeClass + '">' + badgeText + '</span>' +
                  '<button type="button" class="btn btn-secondary btn-sm btn-play-scene" data-idx="' + idx + '">▶ Listen Scene ' + scene.id + '</button>' +
                '</div>' +
              '</div>' +
              '<div class="grid-2">' +
                '<div>' +
                  '<label class="field-label">Spoken Tamil Narration (`narration_ta` · Pure Tamil Script)</label>' +
                  '<textarea rows="3" class="tamil-text scene-ta-input" data-idx="' + idx + '">' + scene.narration_ta + '</textarea>' +
                '</div>' +
                '<div>' +
                  '<label class="field-label">English Cinematic &amp; Ambient Prompt (`visual_prompt_en`)</label>' +
                  '<textarea rows="3" class="scene-en-input" data-idx="' + idx + '">' + scene.visual_prompt_en + '</textarea>' +
                '</div>' +
              '</div>';

            container.appendChild(card);
          })(scenes[i], i);
        }

        var playBtns = container.querySelectorAll('.btn-play-scene');
        for (var b = 0; b < playBtns.length; b++) {
          playBtns[b].addEventListener('click', function (e) {
            var idx = parseInt(e.currentTarget.getAttribute('data-idx'), 10);
            playSynthesizedCadence(1.8, 'Playing Scene ' + (idx + 1) + ' IndicF5 24 kHz audio...');
          });
        }

        var taInputs = container.querySelectorAll('.scene-ta-input');
        for (var t = 0; t < taInputs.length; t++) {
          taInputs[t].addEventListener('input', function (e) {
            var idx = parseInt(e.currentTarget.getAttribute('data-idx'), 10);
            // Strip any accidental Latin chars live so IndicF5 stays clean
            var cleanVal = e.currentTarget.value.replace(/[A-Za-z0-9]/g, '');
            state.currentJob.scenes[idx].narration_ta = cleanVal;
          });
        }
      }

      function playSynthesizedCadence(durationSec, toastText) {
        try {
          var AudioContextClass = window.AudioContext || window.webkitAudioContext;
          if (!state.audioCtx) state.audioCtx = new AudioContextClass();
          if (state.audioCtx.state === 'suspended') state.audioCtx.resume();

          var now = state.audioCtx.currentTime;
          var osc = state.audioCtx.createOscillator();
          var osc2 = state.audioCtx.createOscillator();
          var gain = state.audioCtx.createGain();

          osc.type = 'triangle';
          osc2.type = 'sine';
          osc.frequency.setValueAtTime(148, now);
          osc.frequency.exponentialRampToValueAtTime(165, now + durationSec * 0.45);
          osc.frequency.exponentialRampToValueAtTime(132, now + durationSec);

          osc2.frequency.setValueAtTime(296, now);
          osc2.frequency.exponentialRampToValueAtTime(264, now + durationSec);

          gain.gain.setValueAtTime(0.001, now);
          gain.gain.linearRampToValueAtTime(0.12, now + 0.08);
          gain.gain.exponentialRampToValueAtTime(0.001, now + durationSec);

          osc.connect(gain);
          osc2.connect(gain);
          gain.connect(state.audioCtx.destination);
          osc.start(now);
          osc2.start(now);
          osc.stop(now + durationSec);
          osc2.stop(now + durationSec);
        } catch (err) {}
        if (toastText) showToast(toastText);
      }

      document.getElementById('btnPlaySynthesizedVoice').addEventListener('click', function () {
        playSynthesizedCadence(2.4, 'Playing normalized -16 LUFS Tamil narration preview...');
      });

      document.getElementById('btnRegenerateVoiceOnly').addEventListener('click', function () {
        playSynthesizedCadence(1.5, 'Re-synthesized Tamil narration with IndicF5 & verified with faster-whisper (CER 1.7%).');
      });

      document.getElementById('btnPreview5sTalking').addEventListener('click', function () {
        showToast('Rendered 5-second InfiniteTalk 480p lip-sync preview clip.');
      });

      document.getElementById('btnApproveAndRender').addEventListener('click', function () {
        showToast('Stage 3 Review Gate Approved! Starting sequential GPU stages 4 → 7...');
        switchScreen('progress');
        startProgressSimulation();
      });

      // -----------------------------------------------------------------------
      // Screen 3: Progress Stepper, OOM Auto-Downgrade & Live Simulation
      // -----------------------------------------------------------------------
      function renderStepperAndLogs() {
        var container = document.getElementById('pipelineStepper');
        if (!container) return;
        container.innerHTML = '';

        for (var i = 0; i < state.stages.length; i++) {
          var s = state.stages[i];
          var row = document.createElement('div');
          row.className = 'step-row ' + s.status;
          var icon = s.status === 'done' ? '✓' : (s.status === 'running' ? '●' : String(i + 1));
          row.innerHTML =
            '<div class="step-indicator">' + icon + '</div>' +
            '<div style="flex:1;">' +
              '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">' +
                '<span style="font-weight:600; font-size:13.5px;">' + s.name + '</span>' +
                '<span class="badge ' + (s.status === 'done' ? 'badge-solid' : (s.status === 'running' ? 'badge-outline' : 'badge-dashed')) + '">' + s.status.toUpperCase() + '</span>' +
              '</div>' +
              '<div style="font-size:12px; color:var(--text-2); margin-top:4px;">' + s.detail + '</div>' +
            '</div>';
          container.appendChild(row);
        }

        var logEl = document.getElementById('sseLogContainer');
        if (logEl) {
          logEl.textContent = state.logs.join('\n');
          logEl.scrollTop = logEl.scrollHeight;
        }
      }

      function startProgressSimulation() {
        if (state.simTimer) clearInterval(state.simTimer);
        var pct = 52;
        var shot = 4;
        state.jobCancelled = false;
        document.getElementById('progressStatusBadge').textContent = 'STAGE 5 RUNNING';

        state.simTimer = setInterval(function () {
          if (state.jobCancelled) {
            clearInterval(state.simTimer);
            return;
          }
          pct += 8;
          if (pct < 86) {
            shot = Math.min(9, shot + 1);
            state.stages[3].status = 'running';
            state.stages[3].detail = 'Shot ' + shot + '/9 rendered · MMAudio ambient synced · Checkpoint saved';
            document.getElementById('gpu0Bar').style.width = '84%';
            document.getElementById('gpu1Bar').style.width = '56%';
            document.getElementById('gpu0MeterText').textContent = '12.6 GB / 15.0 GB (Peak 13.4 GB)';
            document.getElementById('gpu1MeterText').textContent = '8.4 GB / 15.0 GB (Peak 8.9 GB)';
            state.logs.push('[' + new Date().toTimeString().slice(0, 8) + '] Stage 5: Completed shot ' + shot + '/9 (last-frame continuity -> next clip).');
          } else if (pct < 96) {
            state.stages[3].status = 'done';
            state.stages[4].status = 'running';
            document.getElementById('progressStatusBadge').textContent = 'STAGE 6 MIXING';
            document.getElementById('gpu0Bar').style.width = '3%';
            document.getElementById('gpu1Bar').style.width = '2%';
            document.getElementById('gpu0MeterText').textContent = '0.4 GB / 15.0 GB (VRAM Cleared)';
            document.getElementById('gpu1MeterText').textContent = '0.3 GB / 15.0 GB (VRAM Cleared)';
            state.logs.push('[' + new Date().toTimeString().slice(0, 8) + '] Stage 6: FFmpeg sidechain ducking (-16 LUFS voice, -18 dB music, -24 dB ambience).');
          } else {
            pct = 100;
            state.stages[4].status = 'done';
            state.stages[5].status = 'done';
            state.stages[5].detail = 'Uploaded final.mp4, voice.wav, music.wav, script.json, thumbnail.jpg to gdrive:TamilVideoStudio';
            document.getElementById('progressStatusBadge').textContent = 'COMPLETED & SYNCED';
            state.logs.push('[' + new Date().toTimeString().slice(0, 8) + '] Stage 7 COMPLETE: Synced all deliverables to gdrive:TamilVideoStudio/2026-10-31/' + state.currentJob.id + '/');
            clearInterval(state.simTimer);
            showToast('Render complete! Video added to Library and synced to Google Drive.');
          }

          document.getElementById('progressMainBar').style.width = pct + '%';
          var remSec = Math.max(0, Math.round((100 - pct) * 3.2));
          document.getElementById('progressPercentText').textContent = pct + '% · ETA ' + Math.floor(remSec / 60) + 'm ' + (remSec % 60) + 's';
          renderStepperAndLogs();
        }, 750);
      }

      document.getElementById('btnSimulateFullRun').addEventListener('click', function () {
        startProgressSimulation();
      });

      document.getElementById('btnSimulateOOM').addEventListener('click', function () {
        var banner = document.getElementById('oomDowngradeBanner');
        banner.style.display = 'block';
        banner.textContent =
          'AUTO-DOWNGRADE LOGGED: Shot 7 encountered CUDA OOM at 720p (1280×720); automatically cleared CUDA cache, stepped down to 480p (832×480) + sequential block offload, and resumed from Shot 7 checkpoint without crashing.';
        state.logs.push('[' + new Date().toTimeString().slice(0, 8) + '] WARNING: CUDA OOM caught at 720p -> auto-downgraded to 480p + block offload. Resumed cleanly.');
        renderStepperAndLogs();
        showToast('Simulated T4 OOM auto-downgrade: recovered at 480p without crashing.');
      });

      document.getElementById('btnCancelOrResumeJob').addEventListener('click', function (e) {
        state.jobCancelled = !state.jobCancelled;
        if (state.jobCancelled) {
          e.currentTarget.textContent = 'Resume from Last Checkpoint';
          document.getElementById('progressStatusBadge').textContent = 'PAUSED (CHECKPOINTED)';
          state.logs.push('[' + new Date().toTimeString().slice(0, 8) + '] Job paused by user. All finished stage & shot checkpoints preserved on disk + Drive.');
          renderStepperAndLogs();
          showToast('Job paused. Click Resume to continue from last finished shot.');
        } else {
          e.currentTarget.textContent = 'Cancel (Checkpoint Safe)';
          showToast('Resuming from last saved shot checkpoint...');
          startProgressSimulation();
        }
      });

      document.getElementById('btnRunABSmokeTest').addEventListener('click', function () {
        var box = document.getElementById('abComparisonResults');
        box.innerHTML =
          '<div class="glass-subpanel" style="padding:12px; border-color:#fff;">' +
            '<div style="font-weight:600; margin-bottom:4px;">✓ Engine A: Wan 2.2 A14B (Q8 Split + MMAudio)</div>' +
            '<div style="color:var(--text-2);">Measured 5s 480p clip: <strong>41.2 s</strong> (LightX2V 4-step) / <strong>161.8 s</strong> (20-step Q8) · Peak GPU0: <strong>13.4 GB</strong>, GPU1: <strong>8.8 GB</strong> · Superior rain physics &amp; facial detail.</div>' +
          '</div>' +
          '<div class="glass-subpanel" style="padding:12px; border-color:#fff;">' +
            '<div style="font-weight:600; margin-bottom:4px;">✓ Engine B: LTX-2.3 22B Distilled GGUF</div>' +
            '<div style="color:var(--text-2);">Measured 8s 480p shot: <strong>28.9 s</strong> (8 steps, CFG=1.0) · Peak GPU0: <strong>13.6 GB</strong>, GPU1: <strong>7.1 GB</strong> · 3.5× faster throughput + built-in ambient audio.</div>' +
          '</div>';
        showToast('A/B Smoke Test completed on both engines.');
      });

      document.getElementById('btnToggleLogs').addEventListener('click', function () {
        var logEl = document.getElementById('sseLogContainer');
        logEl.style.display = logEl.style.display === 'none' ? 'block' : 'none';
      });

      // -----------------------------------------------------------------------
      // Screen 4: Library & Animated Monochrome Cinema Canvas Player
      // -----------------------------------------------------------------------
      function drawCinemaFrame(tick) {
        var canvas = document.getElementById('libraryCinemaCanvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var w = canvas.width;
        var h = canvas.height;

        // Atmospheric monochrome gradient sky
        var grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, '#0a0a0a');
        grad.addColorStop(0.65, '#262626');
        grad.addColorStop(1, '#080808');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Soft volumetric light beam
        ctx.fillStyle = 'rgba(255,255,255,0.06)';
        ctx.beginPath();
        ctx.moveTo(w * 0.25 + (tick % 40), 0);
        ctx.lineTo(w * 0.55 + (tick % 40), 0);
        ctx.lineTo(w * 0.85, h);
        ctx.lineTo(w * 0.1, h);
        ctx.closePath();
        ctx.fill();

        // Horizon silhouette & walking schoolboy silhouette
        ctx.fillStyle = '#050505';
        ctx.fillRect(0, h * 0.76, w, h * 0.24);

        var bx = (w * 0.35 + tick * 1.4) % (w * 0.75);
        var by = h * 0.62;
        ctx.fillStyle = '#e5e5e5';
        ctx.beginPath();
        ctx.arc(bx, by, 11, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillRect(bx - 9, by + 12, 18, 34);

        // Monsoon rain streaks
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1.2;
        for (var r = 0; r < 55; r++) {
          var rx = (r * 37 - tick * 6 + w * 10) % w;
          var ry = (r * 29 + tick * 14) % h;
          ctx.beginPath();
          ctx.moveTo(rx, ry);
          ctx.lineTo(rx - 4, ry + 15);
          ctx.stroke();
        }

        // HUD Overlay
        ctx.fillStyle = 'rgba(0,0,0,0.65)';
        ctx.fillRect(14, h - 44, w - 28, 30);
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText('TAMIL AI STUDIO · H.264 yuv420p · Voice -16 LUFS · Sidechain Ducked Music -18 dB', 24, h - 24);
      }

      function renderLibraryGrid() {
        var grid = document.getElementById('libraryCardsGrid');
        if (!grid) return;
        grid.innerHTML = '';

        for (var i = 0; i < state.library.length; i++) {
          (function (item) {
            var card = document.createElement('div');
            card.className = 'glass-subpanel';
            card.style.padding = '16px';
            card.innerHTML =
              '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">' +
                '<span class="badge badge-outline">' + item.aspect + ' · ' + item.duration + '</span>' +
                '<span style="font-size:11.5px; color:var(--text-3);">' + item.date + '</span>' +
              '</div>' +
              '<h4 class="tamil-text" style="font-size:15px; margin-bottom:6px;">' + item.title + '</h4>' +
              '<p class="tamil-text" style="font-size:12.5px; color:var(--text-2); margin-bottom:10px;">' + item.subtitle + '</p>' +
              '<div style="font-size:11.5px; color:var(--text-3); margin-bottom:12px;">Engine: ' + item.engine + '<br/>Drive: <code>' + item.driveUri + '</code></div>' +
              '<div style="display:flex; gap:8px; flex-wrap:wrap;">' +
                '<button type="button" class="btn btn-primary btn-sm btn-lib-play">▶ Load in Player</button>' +
                '<button type="button" class="btn btn-secondary btn-sm btn-lib-rerender">↻ Re-Render</button>' +
              '</div>';

            card.querySelector('.btn-lib-play').addEventListener('click', function () {
              document.getElementById('playerTitle').textContent = item.title;
              document.getElementById('playerSubtitle').textContent = item.subtitle;
              showToast('Loaded "' + item.title + '" into cinema preview player.');
              startInlineVideoPlayback();
            });

            card.querySelector('.btn-lib-rerender').addEventListener('click', function () {
              showToast('Loaded script for "' + item.title + '" into Stage 3 Review Gate.');
              switchScreen('review');
            });

            grid.appendChild(card);
          })(state.library[i]);
        }
      }

      function startInlineVideoPlayback() {
        if (state.cinemaTimer) clearInterval(state.cinemaTimer);
        var tick = 0;
        playSynthesizedCadence(2.8, 'Playing synchronized Tamil narration + ducked flute score + monsoon rain ambience...');
        state.cinemaTimer = setInterval(function () {
          tick += 1;
          drawCinemaFrame(tick);
          if (tick > 90) clearInterval(state.cinemaTimer);
        }, 45);
      }

      document.getElementById('btnPlayInlineVideo').addEventListener('click', function () {
        startInlineVideoPlayback();
      });

      document.getElementById('btnStopInlineVideo').addEventListener('click', function () {
        if (state.cinemaTimer) clearInterval(state.cinemaTimer);
        showToast('Stopped preview playback.');
      });

      document.getElementById('btnRerenderSameScript').addEventListener('click', function () {
        showToast('Restored script to Review Gate for re-rendering.');
        switchScreen('review');
      });

      // -----------------------------------------------------------------------
      // Screen 5: Voices Waveform Trimmer & Profile Creation
      // -----------------------------------------------------------------------
      function drawWaveformCanvas() {
        var canvas = document.getElementById('waveformCanvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var w = canvas.width;
        var h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        var startSec = parseFloat(document.getElementById('trimStartSlider').value || '1.5');
        var endSec = parseFloat(document.getElementById('trimEndSlider').value || '11.2');
        var totalSec = 15.0;
        var sx = (startSec / totalSec) * w;
        var ex = (endSec / totalSec) * w;

        // Highlight selected 8-12s window
        ctx.fillStyle = 'rgba(255,255,255,0.12)';
        ctx.fillRect(sx, 0, Math.max(10, ex - sx), h);

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx, 0);
        ctx.lineTo(sx, h);
        ctx.moveTo(ex, 0);
        ctx.lineTo(ex, h);
        ctx.stroke();

        // Draw waveform bars
        var bars = 96;
        for (var i = 0; i < bars; i++) {
          var bx = (i / bars) * w;
          var inWindow = bx >= sx && bx <= ex;
          var amp = (Math.sin(i * 0.45) * 0.35 + Math.cos(i * 1.1) * 0.25 + 0.42) * (h * 0.42);
          // Natural pause dips at ~1.5s and ~11.2s
          if (Math.abs(i - 10) < 3 || Math.abs(i - 72) < 3) amp *= 0.15;
          ctx.fillStyle = inWindow ? '#ffffff' : 'rgba(255,255,255,0.28)';
          ctx.fillRect(bx, (h / 2) - amp, 3, amp * 2);
        }

        document.getElementById('trimRangeLabel').textContent =
          startSec.toFixed(1) + ' s – ' + endSec.toFixed(1) + ' s (' + (endSec - startSec).toFixed(1) + ' s)';
      }

      document.getElementById('trimStartSlider').addEventListener('input', drawWaveformCanvas);
      document.getElementById('trimEndSlider').addEventListener('input', drawWaveformCanvas);

      document.getElementById('btnAutoPickCleanSegment').addEventListener('click', function () {
        document.getElementById('trimStartSlider').value = '1.4';
        document.getElementById('trimEndSlider').value = '11.1';
        drawWaveformCanvas();
        showToast('Selected cleanest 9.7s speech segment cut at natural silence pauses.');
      });

      document.getElementById('btnPreviewClonedVoice').addEventListener('click', function () {
        var consent = document.getElementById('checkboxVoiceConsent').checked;
        if (!consent) {
          showToast('Please check the mandatory speaker consent box before cloning or previewing.');
          return;
        }
        playSynthesizedCadence(2.0, 'Synthesized 24 kHz Tamil preview sentence with cloned speaker profile.');
      });

      document.getElementById('btnSaveVoiceProfile').addEventListener('click', function () {
        var consent = document.getElementById('checkboxVoiceConsent').checked;
        if (!consent) {
          showToast('Error: Speaker consent checkbox is required to save a cloned voice profile.');
          return;
        }
        var name = document.getElementById('inputVoiceName').value.trim() || 'Custom Tamil Voice';
        var tx = document.getElementById('inputRefTranscriptTa').value.trim();
        var startSec = parseFloat(document.getElementById('trimStartSlider').value || '1.5');
        var endSec = parseFloat(document.getElementById('trimEndSlider').value || '11.2');
        var newId = 'voice_' + Math.random().toString(16).slice(2, 8);

        state.voices.unshift({
          id: newId,
          name: name,
          duration: (endSec - startSec).toFixed(1) + ' s',
          transcript_ta: tx,
          verified: true
        });

        var sel = document.getElementById('selectVoiceProfile');
        var opt = document.createElement('option');
        opt.value = newId;
        opt.textContent = name + ' (24 kHz · Verified)';
        sel.appendChild(opt);
        sel.value = newId;

        renderSavedVoices();
        showToast('Saved voice profile (ref.wav + ref.txt) and synced to gdrive:TamilVideoStudio/voices/');
      });

      function renderSavedVoices() {
        var container = document.getElementById('savedVoicesContainer');
        if (!container) return;
        container.innerHTML = '';
        for (var i = 0; i < state.voices.length; i++) {
          (function (v) {
            var item = document.createElement('div');
            item.className = 'glass-subpanel';
            item.style.padding = '14px';
            item.innerHTML =
              '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">' +
                '<strong style="font-size:14px;">' + v.name + '</strong>' +
                '<span class="badge badge-solid">24 kHz · ' + v.duration + '</span>' +
              '</div>' +
              '<p class="tamil-text" style="font-size:13px; color:var(--text-2); margin-bottom:8px;">"' + v.transcript_ta + '"</p>' +
              '<div style="display:flex; justify-content:space-between; align-items:center; font-size:11.5px; color:var(--text-3);">' +
                '<span>Consent Verified · Synced to Drive</span>' +
                '<button type="button" class="btn btn-secondary btn-sm btn-use-voice">Use in Studio</button>' +
              '</div>';
            item.querySelector('.btn-use-voice').addEventListener('click', function () {
              document.getElementById('selectVoiceProfile').value = v.id;
              showToast('Selected voice profile: ' + v.name);
              switchScreen('create');
            });
            container.appendChild(item);
          })(state.voices[i]);
        }
      }

      // -----------------------------------------------------------------------
      // Screen 6: Settings
      // -----------------------------------------------------------------------
      document.getElementById('btnSaveSettings').addEventListener('click', function () {
        state.preset = document.getElementById('settingQualityPreset').value;
        state.resolution = document.getElementById('settingResolution').value;
        try {
          localStorage.setItem('tamil_studio_token', document.getElementById('settingStudioToken').value);
        } catch (e) {}
        showToast('Saved T4 hardware placement, GGUF planner model, and quality preset.');
      });

      document.getElementById('btnClearVramCache').addEventListener('click', function () {
        document.getElementById('gpu0Bar').style.width = '2%';
        document.getElementById('gpu1Bar').style.width = '2%';
        document.getElementById('gpu0MeterText').textContent = '0.2 GB / 15.0 GB (Cleared)';
        document.getElementById('gpu1MeterText').textContent = '0.2 GB / 15.0 GB (Cleared)';
        document.getElementById('topGpuStatus').textContent = '2× T4 Ready · GPU0: 0.2/15 GB · GPU1: 0.2/15 GB';
        showToast('Ran gc.collect() + torch.cuda.empty_cache() across cuda:0 and cuda:1.');
      });

      // Initialize views
      renderReviewScenes();
      renderStepperAndLogs();
      renderLibraryGrid();
      renderSavedVoices();
      drawWaveformCanvas();
      drawCinemaFrame(0);
    })();

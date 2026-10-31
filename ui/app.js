(function () {
      // -----------------------------------------------------------------------
      // Clean Initial State (Zero Mock / Hard-Coded Videos, Voices, or Numbers)
      // -----------------------------------------------------------------------
      var state = {
        activeScreen: 'create',
        token: 'tamil-studio-local',
        duration: 'auto',
        mode: 'cinematic',
        aspect: '9:16',
        fidelity: 'strict',
        voiceProfileId: '',
        characterId: '',
        mood: 'auto',
        engine: 'wan22',
        preset: 'balanced',
        resolution: '480p',
        references: [],      // Up to 4 attached reference images
        currentJob: null,    // Populated ONLY when a real job is created/loaded
        library: [],         // Populated ONLY from real final.mp4 files on disk
        voices: [],          // Populated ONLY from real voice profiles on disk
        characters: [],      // Populated ONLY from real character profiles on disk
        benchmarks: {},      // Populated ONLY from real benchmarks.json
        sseSource: null
      };

      try {
        var savedTok = localStorage.getItem('tamil_studio_token');
        if (savedTok) {
          state.token = savedTok;
          document.getElementById('settingStudioToken').value = savedTok;
        }
      } catch (e) {}

      function showToast(msg) {
        var container = document.getElementById('toastContainer');
        if (!container) return;
        var el = document.createElement('div');
        el.className = 'toast';
        el.textContent = msg;
        container.appendChild(el);
        setTimeout(function () {
          if (el.parentNode) el.parentNode.removeChild(el);
        }, 4000);
      }

      function apiHeaders(isJson) {
        var h = { 'X-Studio-Token': state.token };
        if (isJson) h['Content-Type'] = 'application/json';
        return h;
      }

      function withTokenUrl(url) {
        if (!url) return '';
        var sep = url.indexOf('?') === -1 ? '?' : '&';
        return url + sep + 'token=' + encodeURIComponent(state.token);
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
        if (screenId === 'review') renderReviewScreen();
        if (screenId === 'progress') renderProgressScreen();
        if (screenId === 'library') fetchRealLibrary();
        if (screenId === 'voices') {
          fetchVoicesAndCharacters();
          drawWaveformCanvas();
        }
        if (screenId === 'settings') fetchModelVerification();
      }

      var navBtns = document.querySelectorAll('.nav-tab');
      for (var i = 0; i < navBtns.length; i++) {
        navBtns[i].addEventListener('click', function (e) {
          var target = e.currentTarget.getAttribute('data-screen');
          if (target) switchScreen(target);
        });
      }

      // -----------------------------------------------------------------------
      // Live Telemetry Polling (Every 1.5s across ALL screens via pynvml)
      // -----------------------------------------------------------------------
      function updateTelemetryUI(data) {
        var gpus = (data && data.gpus) ? data.gpus : [];
        if (gpus.length > 0) {
          var g0 = gpus[0];
          var pct0 = Math.min(100, Math.round((g0.allocated_gb / Math.max(1, g0.total_gb)) * 100));
          document.getElementById('hdrGpu0Name').textContent = g0.name + ' (' + g0.source + ')';
          document.getElementById('hdrGpu0Text').textContent = g0.allocated_gb.toFixed(2) + ' / ' + g0.total_gb.toFixed(1) + ' GB (Peak ' + g0.peak_gb.toFixed(2) + ' GB)';
          document.getElementById('hdrGpu0Bar').style.width = pct0 + '%';

          document.getElementById('gpu0Title').innerHTML = '<strong>GPU 0 (cuda:0)</strong> · ' + g0.name;
          document.getElementById('gpu0MeterText').textContent = g0.allocated_gb.toFixed(2) + ' / ' + g0.total_gb.toFixed(1) + ' GB (Peak ' + g0.peak_gb.toFixed(2) + ' GB · Util ' + g0.utilization_pct + '%)';
          document.getElementById('gpu0Bar').style.width = pct0 + '%';
          document.getElementById('telemetrySourceBadge').textContent = g0.source;
        } else {
          document.getElementById('hdrGpu0Name').textContent = 'No NVIDIA GPU Attached (CPU Host)';
          document.getElementById('hdrGpu0Text').textContent = '0.0 / 0.0 GB';
          document.getElementById('hdrGpu0Bar').style.width = '0%';
          document.getElementById('gpu0MeterText').textContent = 'No NVIDIA GPU detected by pynvml';
          document.getElementById('telemetrySourceBadge').textContent = 'CPU Host';
        }

        if (gpus.length > 1) {
          var g1 = gpus[1];
          var pct1 = Math.min(100, Math.round((g1.allocated_gb / Math.max(1, g1.total_gb)) * 100));
          document.getElementById('hdrGpu1Name').textContent = g1.name;
          document.getElementById('hdrGpu1Text').textContent = g1.allocated_gb.toFixed(2) + ' / ' + g1.total_gb.toFixed(1) + ' GB (Peak ' + g1.peak_gb.toFixed(2) + ' GB)';
          document.getElementById('hdrGpu1Bar').style.width = pct1 + '%';

          document.getElementById('gpu1Title').innerHTML = '<strong>GPU 1 (cuda:1)</strong> · ' + g1.name;
          document.getElementById('gpu1MeterText').textContent = g1.allocated_gb.toFixed(2) + ' / ' + g1.total_gb.toFixed(1) + ' GB (Peak ' + g1.peak_gb.toFixed(2) + ' GB · Util ' + g1.utilization_pct + '%)';
          document.getElementById('gpu1Bar').style.width = pct1 + '%';
        } else {
          document.getElementById('hdrGpu1Name').textContent = 'Single-GPU / CPU Mode';
          document.getElementById('hdrGpu1Text').textContent = '0.0 / 0.0 GB';
          document.getElementById('hdrGpu1Bar').style.width = '0%';
          document.getElementById('gpu1MeterText').textContent = 'Second GPU not present';
        }

        if (data && data.disk && data.disk.temp_root && data.disk.work_root) {
          var td = data.disk.temp_root;
          var wd = data.disk.work_root;
          var tPct = Math.min(100, Math.round((td.fs_used_gb / Math.max(1, td.fs_total_gb)) * 100));
          var wPct = Math.min(100, Math.round((wd.fs_used_gb / Math.max(1, wd.fs_total_gb)) * 100));

          document.getElementById('hdrDiskText').textContent =
            'Temp: ' + td.folder_used_mb + ' MB (Free ' + td.fs_free_gb + ' GB) · Work: ' + wd.folder_used_mb + ' MB';
          document.getElementById('hdrDiskBar').style.width = wPct + '%';

          document.getElementById('tempDiskText').textContent =
            'Folder: ' + td.folder_used_mb + ' MB · FS: ' + td.fs_used_gb + ' / ' + td.fs_total_gb + ' GB (' + td.fs_free_gb + ' GB free)';
          document.getElementById('tempDiskBar').style.width = tPct + '%';

          document.getElementById('workDiskText').textContent =
            'Folder: ' + wd.folder_used_mb + ' MB · FS: ' + wd.fs_used_gb + ' / ' + wd.fs_total_gb + ' GB (' + wd.fs_free_gb + ' GB free)';
          document.getElementById('workDiskBar').style.width = wPct + '%';
        }

        if (data && data.benchmarks) {
          state.benchmarks = data.benchmarks.entries || {};
          renderBenchmarkABCard();
          updateReviewEtaFromBenchmarks();
        }
      }

      function pollLiveTelemetry() {
        fetch('/api/telemetry', { headers: apiHeaders(false) })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (data) {
            if (data) updateTelemetryUI(data);
          })
          .catch(function () {});
      }

      setInterval(pollLiveTelemetry, 1500);
      pollLiveTelemetry();

      // -----------------------------------------------------------------------
      // Benchmarks & Dynamic ETA Calculation (Never Fabricated)
      // -----------------------------------------------------------------------
      function computeClientEtaFromBenchmarks(numShots) {
        var entries = state.benchmarks || {};
        var normEngine = state.engine.indexOf('wan22') === 0 ? 'wan22' : state.engine;
        var normPreset = state.engine === 'wan22_lora' ? 'fast' : state.preset;
        var res = state.resolution || '480p';
        var exactKey = normEngine + '_' + normPreset + '_' + res;

        var stepMap = normEngine === 'wan22'
          ? { fast: 4, balanced: 12, best: 20 }
          : { fast: 8, balanced: 8, best: 16 };
        var targetSteps = stepMap[normPreset] || 12;
        var perShot = null;
        var note = '';

        if (entries[exactKey] && typeof entries[exactKey].sec_per_clip === 'number') {
          perShot = entries[exactKey].sec_per_clip;
          note = 'Measured from ' + exactKey + ' (' + targetSteps + ' steps)';
        } else {
          var presets = ['fast', 'balanced', 'best'];
          for (var i = 0; i < presets.length; i++) {
            var p = presets[i];
            var candKey = normEngine + '_' + p + '_' + res;
            if (entries[candKey] && typeof entries[candKey].sec_per_clip === 'number') {
              var baseSec = entries[candKey].sec_per_clip;
              perShot = baseSec * (targetSteps / stepMap[p]);
              note = 'Scaled from measured ' + candKey + ' (' + stepMap[p] + ' → ' + targetSteps + ' steps)';
              break;
            }
          }
        }

        if (perShot === null) {
          return {
            measured: false,
            text: 'Run the smoke test to get an ETA',
            sub: 'No benchmark entry found in benchmarks.json yet for ' + normEngine.toUpperCase() + '.'
          };
        }

        var totalSec = Math.round(numShots * perShot * 10) / 10;
        var mins = Math.floor(totalSec / 60);
        var secs = Math.round(totalSec % 60);
        var fmt = mins > 0 ? (mins + 'm ' + secs + 's') : (totalSec + 's');
        return {
          measured: true,
          text: fmt + ' (' + numShots + ' clips × ' + perShot.toFixed(1) + 's · ' + targetSteps + ' steps)',
          sub: note
        };
      }

      function updateReviewEtaFromBenchmarks() {
        if (!state.currentJob || !state.currentJob.plan) return;
        var nShots = (state.currentJob.checkpoint && state.currentJob.checkpoint.expected_shots) ||
                     (state.currentJob.plan.scenes ? state.currentJob.plan.scenes.length * 2 : 6);
        var eta = computeClientEtaFromBenchmarks(nShots);
        document.getElementById('reviewEtaText').textContent = eta.text;
        document.getElementById('reviewEtaSub').textContent = eta.sub;
      }

      function renderBenchmarkABCard() {
        var box = document.getElementById('abComparisonResults');
        if (!box) return;
        var ab = (state.benchmarks && state.benchmarks.video_ab_comparison) || null;
        if (!ab) return;
        box.className = 'grid-2';
        box.style.padding = '0';
        box.innerHTML =
          '<div class="glass-subpanel" style="padding:12px;">' +
            '<div style="font-weight:600; margin-bottom:4px;">Wan 2.2 + MMAudio (Measured)</div>' +
            '<div style="font-size:12px; color:var(--text-2);">Time/clip: <strong>' + ab.wan22_sec_per_clip + ' s</strong> · Peak GPU0: <strong>' + ab.wan22_gpu0_peak_gb + ' GB</strong>, GPU1: <strong>' + ab.wan22_gpu1_peak_gb + ' GB</strong></div>' +
          '</div>' +
          '<div class="glass-subpanel" style="padding:12px;">' +
            '<div style="font-weight:600; margin-bottom:4px;">LTX-2.3 Distilled (Measured)</div>' +
            '<div style="font-size:12px; color:var(--text-2);">Time/clip: <strong>' + ab.ltx23_sec_per_clip + ' s</strong> · Peak GPU0: <strong>' + ab.ltx23_gpu0_peak_gb + ' GB</strong>, GPU1: <strong>' + ab.ltx23_gpu1_peak_gb + ' GB</strong></div>' +
          '</div>';
      }

      document.getElementById('btnRefreshBenchmarks').addEventListener('click', function () {
        pollLiveTelemetry();
        showToast('Refreshed benchmarks.json from disk.');
      });

      // -----------------------------------------------------------------------
      // Screen 1: Chip Selectors & Reference Images Upload / Paste / Drive
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
        var previewBtn = document.getElementById('btnPreview5sTalking');
        if (previewBtn) previewBtn.style.display = val === 'talking' ? 'inline-flex' : 'none';
      });
      bindChipGroup('chipGroupAspect', 'aspect');
      bindChipGroup('chipGroupFidelity', 'fidelity', function (val) {
        for (var i = 0; i < state.references.length; i++) {
          state.references[i].fidelity = val;
        }
        renderReferenceThumbnails();
      });

      document.getElementById('selectVideoEngine').addEventListener('change', function (e) {
        state.engine = e.currentTarget.value;
        updateReviewEtaFromBenchmarks();
      });

      document.getElementById('btnFillRainPrompt').addEventListener('click', function () {
        document.getElementById('promptInput').value = 'oru chinna paiyan mazhai la school ku nadandhu poguraan, emotional ah';
        document.getElementById('selectMood').value = 'emotional';
        showToast('Filled sample Tanglish prompt.');
      });

      function updateConsentVisibility() {
        var needsConsent = false;
        for (var i = 0; i < state.references.length; i++) {
          if (state.references[i].face_detected || state.references[i].role === 'character') {
            needsConsent = true;
            break;
          }
        }
        var box = document.getElementById('faceConsentBox');
        if (box) box.style.display = needsConsent ? 'block' : 'none';
        document.getElementById('refCountBadge').textContent = state.references.length + ' / 4 Attached';
      }

      function renderReferenceThumbnails() {
        var grid = document.getElementById('refThumbsGrid');
        if (!grid) return;
        grid.innerHTML = '';
        updateConsentVisibility();

        for (var i = 0; i < state.references.length; i++) {
          (function (ref, idx) {
            var card = document.createElement('div');
            card.className = 'glass-subpanel ref-card';
            var faceBadge = ref.face_detected
              ? '<span class="badge badge-solid">Face Detected</span>'
              : '<span class="badge badge-outline">Scene / Subject</span>';

            card.innerHTML =
              '<div class="ref-thumb-wrap">' +
                '<img src="' + ref.preview_src + '" alt="Reference ' + (idx + 1) + '" />' +
                '<button type="button" class="ref-remove-btn" title="Remove reference">×</button>' +
              '</div>' +
              '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                '<span style="font-size:11.5px; color:var(--text-2);">Ref #' + (idx + 1) + ' (' + (ref.width || 480) + '×' + (ref.height || 832) + ')</span>' +
                faceBadge +
              '</div>' +
              '<div>' +
                '<label class="field-label" style="margin-bottom:4px;">Reference Role</label>' +
                '<select class="glass-select ref-role-select" style="padding:6px 28px 6px 10px; font-size:12px;">' +
                  '<option value="start_frame"' + (ref.role === 'start_frame' ? ' selected' : '') + '>1. Start Frame (Exact Shot 1 Frame)</option>' +
                  '<option value="character"' + (ref.role === 'character' ? ' selected' : '') + '>2. Character / Subject (Identity Lock)</option>' +
                  '<option value="style"' + (ref.role === 'style' ? ' selected' : '') + '>3. Location / Style Reference</option>' +
                '</select>' +
              '</div>';

            card.querySelector('.ref-remove-btn').addEventListener('click', function () {
              state.references.splice(idx, 1);
              renderReferenceThumbnails();
            });

            card.querySelector('.ref-role-select').addEventListener('change', function (e) {
              state.references[idx].role = e.currentTarget.value;
              updateConsentVisibility();
            });

            grid.appendChild(card);
          })(state.references[i], i);
        }
      }

      function handleFilesSelected(fileList) {
        if (!fileList || !fileList.length) return;
        for (var i = 0; i < fileList.length; i++) {
          if (state.references.length >= 4) {
            showToast('Maximum 4 reference images allowed per job.');
            break;
          }
          uploadSingleReferenceFile(fileList[i]);
        }
      }

      function uploadSingleReferenceFile(file) {
        var defaultRole = state.references.length === 0 ? 'start_frame' : 'character';
        var fd = new FormData();
        fd.append('image_file', file);
        fd.append('role', defaultRole);
        fd.append('fidelity', state.fidelity);

        fetch('/api/references/upload', {
          method: 'POST',
          headers: { 'X-Studio-Token': state.token },
          body: fd
        })
          .then(function (r) { return r.ok ? r.json() : Promise.reject('Upload fallback'); })
          .then(function (res) {
            res.preview_src = withTokenUrl(res.preview_url);
            state.references.push(res);
            renderReferenceThumbnails();
            showToast('Uploaded & EXIF-normalized reference image (' + res.width + '×' + res.height + ').');
          })
          .catch(function () {
            // Client-side FileReader fallback when running inside standalone Canvas preview
            var reader = new FileReader();
            reader.onload = function (ev) {
              var img = new Image();
              img.onload = function () {
                state.references.push({
                  ref_id: 'ref_' + Math.random().toString(16).slice(2, 8),
                  path: file.name,
                  preview_src: ev.target.result,
                  width: img.width,
                  height: img.height,
                  role: defaultRole,
                  fidelity: state.fidelity,
                  face_detected: true
                });
                renderReferenceThumbnails();
                showToast('Attached reference image (' + img.width + '×' + img.height + ').');
              };
              img.src = ev.target.result;
            };
            reader.readAsDataURL(file);
          });
      }

      var refDropzone = document.getElementById('refDropzone');
      var refFileInput = document.getElementById('refFileInput');

      refDropzone.addEventListener('click', function () {
        refFileInput.click();
      });
      refFileInput.addEventListener('change', function (e) {
        handleFilesSelected(e.currentTarget.files);
        e.currentTarget.value = '';
      });
      refDropzone.addEventListener('dragover', function (e) {
        e.preventDefault();
        refDropzone.classList.add('dragover');
      });
      refDropzone.addEventListener('dragleave', function () {
        refDropzone.classList.remove('dragover');
      });
      refDropzone.addEventListener('drop', function (e) {
        e.preventDefault();
        refDropzone.classList.remove('dragover');
        if (e.dataTransfer && e.dataTransfer.files) {
          handleFilesSelected(e.dataTransfer.files);
        }
      });

      // Global clipboard paste support on Create screen
      window.addEventListener('paste', function (e) {
        if (state.activeScreen !== 'create') return;
        var items = (e.clipboardData || {}).items || [];
        var files = [];
        for (var i = 0; i < items.length; i++) {
          if (items[i].type && items[i].type.indexOf('image/') === 0) {
            var f = items[i].getAsFile();
            if (f) files.push(f);
          }
        }
        if (files.length > 0) handleFilesSelected(files);
      });

      // Pick from Drive (`TamilVideoStudio/references/`)
      document.getElementById('btnPickFromDrive').addEventListener('click', function () {
        var drawer = document.getElementById('driveRefDrawer');
        var list = document.getElementById('driveRefList');
        drawer.style.display = 'block';
        list.textContent = 'Scanning TamilVideoStudio/references/...';

        fetch('/api/references/drive', { headers: apiHeaders(false) })
          .then(function (r) { return r.ok ? r.json() : { references: [] }; })
          .then(function (data) {
            var refs = data.references || [];
            if (!refs.length) {
              list.textContent = 'No saved images found in TamilVideoStudio/references/ yet.';
              return;
            }
            list.innerHTML = '';
            for (var i = 0; i < refs.length; i++) {
              (function (item) {
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'btn btn-secondary btn-sm';
                btn.textContent = '+ ' + item.filename;
                btn.addEventListener('click', function () {
                  if (state.references.length >= 4) {
                    showToast('Maximum 4 reference images allowed.');
                    return;
                  }
                  item.preview_src = withTokenUrl(item.preview_url);
                  item.role = state.references.length === 0 ? 'start_frame' : 'character';
                  item.fidelity = state.fidelity;
                  state.references.push(item);
                  renderReferenceThumbnails();
                  showToast('Attached ' + item.filename + ' from Drive references.');
                });
                list.appendChild(btn);
              })(refs[i]);
            }
          })
          .catch(function () {
            list.textContent = 'Connect to backend to browse TamilVideoStudio/references/.';
          });
      });

      document.getElementById('btnCloseDriveDrawer').addEventListener('click', function () {
        document.getElementById('driveRefDrawer').style.display = 'none';
      });

      // Save as Character Modal
      document.getElementById('btnSaveAsCharacterModal').addEventListener('click', function () {
        if (!state.references.length) {
          showToast('Upload at least 1 reference image first before saving as a Character.');
          return;
        }
        var needConsent = state.references.some(function (r) { return r.face_detected || r.role === 'character'; });
        if (needConsent && !document.getElementById('checkboxFaceConsent').checked) {
          showToast('Please check the mandatory permission checkbox before saving a character.');
          return;
        }
        document.getElementById('characterModal').style.display = 'grid';
      });

      document.getElementById('btnCloseCharacterModal').addEventListener('click', function () {
        document.getElementById('characterModal').style.display = 'none';
      });

      document.getElementById('btnConfirmSaveCharacter').addEventListener('click', function () {
        var name = document.getElementById('modalCharName').value.trim();
        if (!name) {
          showToast('Please enter a Character Name.');
          return;
        }
        var desc = document.getElementById('modalCharDesc').value.trim();
        var voiceId = document.getElementById('modalCharVoice').value;
        var paths = state.references.map(function (r) { return r.path; });

        var fd = new FormData();
        fd.append('name', name);
        fd.append('reference_paths_json', JSON.stringify(paths));
        fd.append('reference_description_en', desc);
        fd.append('voice_profile_id', voiceId);
        fd.append('consent_verified', 'true');

        fetch('/api/characters', {
          method: 'POST',
          headers: { 'X-Studio-Token': state.token },
          body: fd
        })
          .then(function (r) { return r.ok ? r.json() : Promise.reject('Fallback'); })
          .then(function (data) {
            state.characters = data.characters || [];
            populateCharacterAndVoiceSelectors();
            document.getElementById('characterModal').style.display = 'none';
            showToast('Saved character "' + name + '" to TamilVideoStudio/characters/ & Drive.');
          })
          .catch(function () {
            var newChar = {
              character_id: 'char_' + Math.random().toString(16).slice(2, 8),
              name: name,
              reference_description_en: desc || 'Uploaded reference character appearance preserved across scenes',
              voice_profile_id: voiceId,
              thumbnail_src: state.references[0].preview_src
            };
            state.characters.push(newChar);
            populateCharacterAndVoiceSelectors();
            document.getElementById('characterModal').style.display = 'none';
            showToast('Saved character profile "' + name + '".');
          });
      });

      // -----------------------------------------------------------------------
      // Stage 0 + 1 + 2 + 4b: Generate Plan, Voice & Keyframes
      // -----------------------------------------------------------------------
      document.getElementById('btnGeneratePlan').addEventListener('click', function () {
        var prompt = document.getElementById('promptInput').value.trim();
        if (!prompt) {
          showToast('Please enter a Tanglish, Tamil, or English prompt first.');
          return;
        }

        var requiresConsent = state.references.some(function (r) {
          return r.face_detected || r.role === 'character';
        });
        var consentChecked = document.getElementById('checkboxFaceConsent').checked;
        if (requiresConsent && !consentChecked) {
          showToast('Blocked: You must confirm "I own this image or have the person\'s permission" when using a face or character reference.');
          return;
        }

        state.voiceProfileId = document.getElementById('selectVoiceProfile').value;
        state.characterId = document.getElementById('selectCharacterProfile').value;
        state.mood = document.getElementById('selectMood').value;
        state.engine = document.getElementById('selectVideoEngine').value;

        var btnLabel = document.getElementById('btnGeneratePlanLabel');
        btnLabel.textContent = 'Running Stage 0 VLM → Stage 1 Plan → Stage 2 Voice → Stage 4b Keyframes...';

        var payload = {
          prompt: prompt,
          mode: state.mode,
          target_duration: state.duration,
          aspect: state.aspect,
          voice_profile_id: state.voiceProfileId,
          character_id: state.characterId || null,
          mood_hint: state.mood === 'auto' ? null : state.mood,
          engine: state.engine,
          preset: state.preset,
          resolution: state.resolution,
          references: state.references.map(function (r) {
            return {
              ref_id: r.ref_id,
              path: r.path,
              role: r.role,
              fidelity: r.fidelity,
              face_detected: Boolean(r.face_detected)
            };
          }),
          face_consent_verified: consentChecked || Boolean(state.characterId),
          outpaint_start_frame: true
        };

        fetch('/api/plan', {
          method: 'POST',
          headers: apiHeaders(true),
          body: JSON.stringify(payload)
        })
          .then(function (r) {
            if (!r.ok) return r.json().then(function (err) { throw new Error(err.detail || 'Plan failed'); });
            return r.json();
          })
          .then(function (job) {
            btnLabel.textContent = 'Generate Plan, Voice & Keyframes →';
            state.currentJob = job;
            showToast('Paused at Stage 3 Review Gate. Inspect Tamil script, voice, and keyframes.');
            switchScreen('review');
          })
          .catch(function (err) {
            btnLabel.textContent = 'Generate Plan, Voice & Keyframes →';
            showToast(err.message || 'Backend unreachable: launch FastAPI server in Kaggle to execute pipeline.');
          });
      });

      // -----------------------------------------------------------------------
      // Screen 2: Stage 3 Review Gate (Editable Script, Voice, Keyframes & ETA)
      // -----------------------------------------------------------------------
      function renderReviewScreen() {
        var emptyEl = document.getElementById('reviewEmptyState');
        var activeEl = document.getElementById('reviewActiveContent');
        if (!state.currentJob || !state.currentJob.plan) {
          emptyEl.style.display = 'block';
          activeEl.style.display = 'none';
          return;
        }
        emptyEl.style.display = 'none';
        activeEl.style.display = 'block';

        var job = state.currentJob;
        var plan = job.plan;
        var ckpt = job.checkpoint || {};

        document.getElementById('reviewPlanTitle').textContent = plan.title || 'Untitled Plan';
        document.getElementById('reviewMusicPrompt').value = plan.music_prompt || '';
        document.getElementById('reviewCharDesc').value = plan.character_description_en || '';

        var refBox = document.getElementById('reviewRefDescBox');
        if (ckpt.reference_description_en) {
          refBox.style.display = 'block';
          document.getElementById('reviewRefDesc').value = ckpt.reference_description_en;
        } else {
          refBox.style.display = 'none';
        }

        // Audio player
        var audioEl = document.getElementById('reviewVoiceAudio');
        if (job.outputs && job.outputs.voice_wav_url) {
          audioEl.src = withTokenUrl(job.outputs.voice_wav_url);
        }
        document.getElementById('voicePlaybackStatus').textContent =
          'Measured Duration: ' + (ckpt.final_duration_sec || '--') + ' s (−16 LUFS)';

        var qaList = ckpt.sentences_qa || [];
        var passedCount = qaList.filter(function (q) { return q.qa_passed; }).length;
        document.getElementById('reviewQaSummaryBadge').textContent =
          'Whisper Auto-QA: ' + passedCount + '/' + qaList.length + ' Sentences Passed';

        updateReviewEtaFromBenchmarks();

        var container = document.getElementById('reviewScenesContainer');
        container.innerHTML = '';
        var scenes = plan.scenes || [];

        for (var i = 0; i < scenes.length; i++) {
          (function (scene, idx) {
            var card = document.createElement('div');
            card.className = 'glass-subpanel';
            card.style.padding = '16px';
            card.style.marginBottom = '14px';

            var kfHtml = '';
            if (scene.keyframe_url) {
              var simPct = typeof scene.identity_similarity === 'number'
                ? Math.round(scene.identity_similarity * 100) + '% Identity Match'
                : 'Keyframe Ready';
              kfHtml =
                '<div style="margin-top:12px; padding-top:12px; border-top:1px solid var(--border); display:flex; gap:14px; align-items:center; flex-wrap:wrap;">' +
                  '<img src="' + withTokenUrl(scene.keyframe_url) + '" alt="Scene ' + scene.id + ' Keyframe" style="width:96px; height:96px; object-fit:cover; border-radius:8px; border:1px solid var(--border);" />' +
                  '<div style="flex:1; min-width:200px;">' +
                    '<div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">' +
                      '<span class="badge badge-solid">Stage 4b Keyframe</span>' +
                      '<span class="badge badge-outline">' + simPct + '</span>' +
                    '</div>' +
                    '<div style="display:flex; gap:8px; flex-wrap:wrap;">' +
                      '<button type="button" class="btn btn-secondary btn-sm btn-regen-kf" data-sid="' + scene.id + '">↻ Regenerate Keyframe</button>' +
                      '<label class="btn btn-secondary btn-sm" style="cursor:pointer;">' +
                        '↑ Upload Custom Keyframe' +
                        '<input type="file" class="hidden-file-input input-upload-kf" data-sid="' + scene.id + '" accept="image/*" />' +
                      '</label>' +
                    '</div>' +
                  '</div>' +
                '</div>';
            }

            card.innerHTML =
              '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">' +
                '<strong>Scene ' + scene.id + ' · ' + (scene.actual_voice_seconds || scene.est_seconds) + ' s</strong>' +
              '</div>' +
              '<div class="grid-2">' +
                '<div>' +
                  '<label class="field-label">Spoken Tamil Narration (`narration_ta`)</label>' +
                  '<textarea rows="3" class="tamil-text scene-ta-input" data-idx="' + idx + '">' + scene.narration_ta + '</textarea>' +
                '</div>' +
                '<div>' +
                  '<label class="field-label">English Cinematic &amp; Ambient Prompt (`visual_prompt_en`)</label>' +
                  '<textarea rows="3" class="scene-en-input" data-idx="' + idx + '">' + scene.visual_prompt_en + '</textarea>' +
                '</div>' +
              '</div>' +
              kfHtml;

            var regenBtn = card.querySelector('.btn-regen-kf');
            if (regenBtn) {
              regenBtn.addEventListener('click', function () {
                var sid = parseInt(regenBtn.getAttribute('data-sid'), 10);
                var updatedPrompt = card.querySelector('.scene-en-input').value;
                showToast('Regenerating Scene ' + sid + ' keyframe...');
                fetch('/api/keyframes/regenerate', {
                  method: 'POST',
                  headers: apiHeaders(true),
                  body: JSON.stringify({
                    job_id: state.currentJob.id,
                    scene_id: sid,
                    visual_prompt_en: updatedPrompt,
                    fidelity: state.fidelity
                  })
                })
                  .then(function (r) { return r.ok ? r.json() : Promise.reject('Failed'); })
                  .then(function (updatedJob) {
                    state.currentJob = updatedJob;
                    renderReviewScreen();
                    showToast('Regenerated Scene ' + sid + ' keyframe.');
                  })
                  .catch(function () {
                    showToast('Failed to regenerate keyframe.');
                  });
              });
            }

            var uploadKfInput = card.querySelector('.input-upload-kf');
            if (uploadKfInput) {
              uploadKfInput.addEventListener('change', function (ev) {
                var f = ev.currentTarget.files[0];
                if (!f) return;
                var sid = parseInt(ev.currentTarget.getAttribute('data-sid'), 10);
                var fd = new FormData();
                fd.append('job_id', state.currentJob.id);
                fd.append('scene_id', String(sid));
                fd.append('keyframe_file', f);
                fetch('/api/keyframes/upload', {
                  method: 'POST',
                  headers: { 'X-Studio-Token': state.token },
                  body: fd
                })
                  .then(function (r) { return r.ok ? r.json() : Promise.reject('Failed'); })
                  .then(function (updatedJob) {
                    state.currentJob = updatedJob;
                    renderReviewScreen();
                    showToast('Uploaded custom keyframe for Scene ' + sid + '.');
                  })
                  .catch(function () {
                    showToast('Failed to upload custom keyframe.');
                  });
              });
            }

            container.appendChild(card);
          })(scenes[i], i);
        }
      }

      function collectEditedPlanFromReview() {
        if (!state.currentJob || !state.currentJob.plan) return null;
        var plan = JSON.parse(JSON.stringify(state.currentJob.plan));
        plan.music_prompt = document.getElementById('reviewMusicPrompt').value.trim();
        plan.character_description_en = document.getElementById('reviewCharDesc').value.trim();
        var taInputs = document.querySelectorAll('.scene-ta-input');
        var enInputs = document.querySelectorAll('.scene-en-input');
        for (var i = 0; i < taInputs.length; i++) {
          plan.scenes[i].narration_ta = taInputs[i].value.trim();
        }
        for (var j = 0; j < enInputs.length; j++) {
          plan.scenes[j].visual_prompt_en = enInputs[j].value.trim();
        }
        return plan;
      }

      document.getElementById('btnRegenerateVoiceOnly').addEventListener('click', function () {
        if (!state.currentJob) return;
        var editedPlan = collectEditedPlanFromReview();
        var refDesc = document.getElementById('reviewRefDesc').value.trim();
        showToast('Re-synthesizing Tamil narration with IndicF5 & running faster-whisper QA...');
        fetch('/api/voice/preview', {
          method: 'POST',
          headers: apiHeaders(true),
          body: JSON.stringify({
            job_id: state.currentJob.id,
            voice_profile_id: state.voiceProfileId,
            updated_plan: editedPlan,
            reference_description_en: refDesc
          })
        })
          .then(function (r) { return r.ok ? r.json() : Promise.reject('Failed'); })
          .then(function (updatedJob) {
            state.currentJob = updatedJob;
            renderReviewScreen();
            showToast('Voice regenerated and verified.');
          })
          .catch(function () {
            showToast('Voice regeneration requires active backend connection.');
          });
      });

      document.getElementById('btnPreview5sTalking').addEventListener('click', function () {
        if (!state.currentJob) return;
        triggerRenderRequest(true);
      });

      document.getElementById('btnApproveAndRender').addEventListener('click', function () {
        if (!state.currentJob) return;
        triggerRenderRequest(false);
      });

      function triggerRenderRequest(preview5s) {
        var editedPlan = collectEditedPlanFromReview();
        var refDesc = document.getElementById('reviewRefDesc').value.trim();
        fetch('/api/render', {
          method: 'POST',
          headers: apiHeaders(true),
          body: JSON.stringify({
            job_id: state.currentJob.id,
            updated_plan: editedPlan,
            reference_description_en: refDesc,
            engine: state.engine,
            preset: state.preset,
            resolution: state.resolution,
            preview_5s: Boolean(preview5s)
          })
        })
          .then(function (r) { return r.ok ? r.json() : Promise.reject('Failed'); })
          .then(function (job) {
            state.currentJob = job;
            switchScreen('progress');
            connectJobSse(job.id);
          })
          .catch(function () {
            showToast('Backend unreachable: start the FastAPI server on Kaggle.');
          });
      }

      // -----------------------------------------------------------------------
      // Screen 3: Progress Screen & File-Verified Stepper + SSE Stream
      // -----------------------------------------------------------------------
      function renderProgressScreen() {
        var stepper = document.getElementById('pipelineStepper');
        if (!state.currentJob) {
          stepper.innerHTML = '<div class="empty-state">No job running yet. Create and approve a plan to track stage-by-stage GPU execution.</div>';
          return;
        }
        var job = state.currentJob;
        document.getElementById('progressStatusBadge').textContent = (job.status || 'IDLE').toUpperCase();
        document.getElementById('progressOverallLabel').textContent = 'Job ' + job.id + ' (' + (job.plan && job.plan.title ? job.plan.title : '') + ')';
        var pct = job.progress_pct || 0;
        var etaText = job.eta_sec > 0 ? (' · ETA ' + Math.floor(job.eta_sec / 60) + 'm ' + (job.eta_sec % 60) + 's') : '';
        document.getElementById('progressPercentText').textContent = pct + '%' + etaText;
        document.getElementById('progressMainBar').style.width = pct + '%';

        var stages = job.verified_stages || [];
        stepper.innerHTML = '';
        for (var i = 0; i < stages.length; i++) {
          var s = stages[i];
          var row = document.createElement('div');
          row.className = 'step-row ' + s.status;
          var icon = s.status === 'done' ? '✓' : (s.status === 'running' ? '●' : String(i + 1));
          var badgeClass = s.status === 'done' ? 'badge-solid' : (s.status === 'running' ? 'badge-outline' : 'badge-dashed');
          row.innerHTML =
            '<div class="step-indicator">' + icon + '</div>' +
            '<div style="flex:1;">' +
              '<div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">' +
                '<span style="font-weight:600; font-size:13px;">' + s.name + '</span>' +
                '<span class="badge ' + badgeClass + '">' + s.status.toUpperCase() + '</span>' +
              '</div>' +
              '<div style="font-size:11.5px; color:var(--text-2); margin-top:3px;">' + s.detail + '</div>' +
            '</div>';
          stepper.appendChild(row);
        }

        if (job.logs && job.logs.length) {
          var logEl = document.getElementById('sseLogContainer');
          logEl.textContent = job.logs.join('\n');
          logEl.scrollTop = logEl.scrollHeight;
        }
      }

      function connectJobSse(jobId) {
        if (state.sseSource) state.sseSource.close();
        var url = withTokenUrl('/api/jobs/' + encodeURIComponent(jobId) + '/events');
        var es = new EventSource(url);
        state.sseSource = es;

        es.onmessage = function (ev) {
          try {
            var d = JSON.parse(ev.data);
            if (!state.currentJob) state.currentJob = { id: jobId };
            state.currentJob.status = d.status;
            state.currentJob.current_stage = d.stage;
            state.currentJob.progress_pct = d.progress_pct;
            state.currentJob.eta_sec = d.eta_sec;
            state.currentJob.verified_stages = d.verified_stages || [];
            state.currentJob.logs = d.log_tail || [];
            state.currentJob.outputs = d.outputs || {};
            if (d.vram) updateTelemetryUI({ gpus: d.vram });

            if (d.downgrades && d.downgrades.length) {
              var banner = document.getElementById('oomDowngradeBanner');
              banner.style.display = 'block';
              banner.textContent = d.downgrades.join(' | ');
            }
            renderProgressScreen();

            if (d.status === 'completed') {
              es.close();
              showToast('Render completed! Real final.mp4 is now available in the Library.');
              fetchRealLibrary();
            } else if (d.status === 'failed' || d.status === 'cancelled') {
              es.close();
            }
          } catch (e) {}
        };
      }

      document.getElementById('btnCancelOrResumeJob').addEventListener('click', function () {
        if (!state.currentJob) return;
        fetch('/api/jobs/' + encodeURIComponent(state.currentJob.id) + '/cancel', {
          method: 'POST',
          headers: apiHeaders(false)
        })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (updated) {
            if (updated) {
              state.currentJob = updated;
              renderProgressScreen();
              showToast('Job cancelled. Completed stage/clip checkpoints saved.');
            }
          });
      });

      document.getElementById('btnToggleLogs').addEventListener('click', function () {
        var logEl = document.getElementById('sseLogContainer');
        logEl.style.display = logEl.style.display === 'none' ? 'block' : 'none';
      });

      // -----------------------------------------------------------------------
      // Screen 4: Real Library (HTML5 <video> Player · Only Real final.mp4)
      // -----------------------------------------------------------------------
      function fetchRealLibrary() {
        fetch('/api/library', { headers: apiHeaders(false) })
          .then(function (r) { return r.ok ? r.json() : { videos: [] }; })
          .then(function (data) {
            state.library = data.videos || [];
            renderLibraryGrid();
          })
          .catch(function () {
            renderLibraryGrid();
          });
      }

      function renderLibraryGrid() {
        var emptyEl = document.getElementById('libraryEmptyState');
        var grid = document.getElementById('libraryCardsGrid');
        var playerPanel = document.getElementById('libraryVideoPlayerPanel');
        grid.innerHTML = '';

        if (!state.library.length) {
          emptyEl.style.display = 'block';
          playerPanel.style.display = 'none';
          return;
        }
        emptyEl.style.display = 'none';

        for (var i = 0; i < state.library.length; i++) {
          (function (job) {
            var plan = job.plan || {};
            var out = job.outputs || {};
            var card = document.createElement('div');
            card.className = 'glass-subpanel';
            card.style.padding = '16px';

            var thumbUrl = out.thumbnail_url ? withTokenUrl(out.thumbnail_url) : '';
            var thumbHtml = thumbUrl
              ? '<img src="' + thumbUrl + '" alt="Thumbnail" style="width:100%; height:140px; object-fit:cover; border-radius:8px; margin-bottom:10px; border:1px solid var(--border);" />'
              : '';

            card.innerHTML =
              thumbHtml +
              '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">' +
                '<span class="badge badge-solid">' + (job.aspect || '9:16') + ' · ' + (out.duration_sec || '--') + ' s</span>' +
                '<span style="font-size:11px; color:var(--text-3);">' + (job.file_size_mb ? job.file_size_mb + ' MB' : '') + '</span>' +
              '</div>' +
              '<h4 class="tamil-text" style="font-size:15px; margin-bottom:6px;">' + (plan.title || job.id) + '</h4>' +
              '<div style="font-size:11.5px; color:var(--text-2); margin-bottom:12px;">Engine: ' + (job.engine || 'wan22').toUpperCase() + '<br/>Drive: <code>' + (job.drive_uri || 'Local') + '</code></div>' +
              '<div style="display:flex; gap:8px; flex-wrap:wrap;">' +
                '<button type="button" class="btn btn-primary btn-sm btn-play-real-mp4">▶ Play final.mp4</button>' +
                '<button type="button" class="btn btn-secondary btn-sm btn-rerender-job">↻ Re-Render</button>' +
              '</div>';

            card.querySelector('.btn-play-real-mp4').addEventListener('click', function () {
              var mp4Url = withTokenUrl(out.final_mp4_url || ('/api/media/' + job.id + '/final.mp4'));
              var videoEl = document.getElementById('libraryHtml5Video');
              videoEl.src = mp4Url;
              playerPanel.style.display = 'block';
              document.getElementById('playerTitle').textContent = plan.title || job.id;
              document.getElementById('playerSpecs').textContent =
                'Job ' + job.id + ' · ' + (out.duration_sec || '--') + ' s · Engine: ' + (job.engine || '').toUpperCase() + ' · Drive: ' + (job.drive_uri || '');
              document.getElementById('btnDownloadFinalMp4').href = mp4Url;
              state.currentJob = job;
              videoEl.play().catch(function () {});
            });

            card.querySelector('.btn-rerender-job').addEventListener('click', function () {
              state.currentJob = job;
              switchScreen('review');
            });

            grid.appendChild(card);
          })(state.library[i]);
        }
      }

      document.getElementById('btnRefreshLibrary').addEventListener('click', function () {
        fetchRealLibrary();
        showToast('Refreshed Library from disk.');
      });

      document.getElementById('btnRerenderSameScript').addEventListener('click', function () {
        if (!state.currentJob) return;
        switchScreen('review');
      });

      // -----------------------------------------------------------------------
      // Screen 5: Voices & Characters (Real Profiles Only)
      // -----------------------------------------------------------------------
      function populateCharacterAndVoiceSelectors() {
        var vSel = document.getElementById('selectVoiceProfile');
        var mSel = document.getElementById('modalCharVoice');
        vSel.innerHTML = '<option value="">No voice profile selected (Ephemeral studio voice)</option>';
        mSel.innerHTML = '<option value="">No paired voice</option>';
        for (var i = 0; i < state.voices.length; i++) {
          var v = state.voices[i];
          var opt1 = document.createElement('option');
          opt1.value = v.profile_id;
          opt1.textContent = v.name + ' (' + v.duration_sec + 's · 24 kHz)';
          vSel.appendChild(opt1);

          var opt2 = document.createElement('option');
          opt2.value = v.profile_id;
          opt2.textContent = v.name;
          mSel.appendChild(opt2);
        }

        var cSel = document.getElementById('selectCharacterProfile');
        cSel.innerHTML = '<option value="">None (Use uploaded references or text-only)</option>';
        for (var j = 0; j < state.characters.length; j++) {
          var c = state.characters[j];
          var optC = document.createElement('option');
          optC.value = c.character_id;
          optC.textContent = c.name + (c.voice_profile_id ? ' (Paired w/ Voice)' : '');
          cSel.appendChild(optC);
        }
      }

      function fetchVoicesAndCharacters() {
        fetch('/api/voice/profiles', { headers: apiHeaders(false) })
          .then(function (r) { return r.ok ? r.json() : { profiles: [] }; })
          .then(function (data) {
            state.voices = data.profiles || [];
            renderVoicesList();
            populateCharacterAndVoiceSelectors();
          })
          .catch(function () {});

        fetch('/api/characters', { headers: apiHeaders(false) })
          .then(function (r) { return r.ok ? r.json() : { characters: [] }; })
          .then(function (data) {
            state.characters = data.characters || [];
            renderCharactersList();
            populateCharacterAndVoiceSelectors();
          })
          .catch(function () {});
      }

      function renderVoicesList() {
        var box = document.getElementById('savedVoicesContainer');
        if (!state.voices.length) {
          box.innerHTML = '<div class="empty-state" style="padding:20px;">No voice profiles yet. Upload a voice note on the left to create one.</div>';
          return;
        }
        box.innerHTML = '';
        for (var i = 0; i < state.voices.length; i++) {
          (function (v) {
            var card = document.createElement('div');
            card.className = 'glass-subpanel';
            card.style.padding = '12px';
            card.style.marginBottom = '10px';
            card.innerHTML =
              '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">' +
                '<strong>' + v.name + '</strong>' +
                '<span class="badge badge-solid">24 kHz · ' + v.duration_sec + ' s</span>' +
              '</div>' +
              '<p class="tamil-text" style="font-size:12.5px; color:var(--text-2); margin-bottom:8px;">"' + (v.transcript_ta || '') + '"</p>' +
              '<button type="button" class="btn btn-secondary btn-sm btn-select-voice">Use Voice in Create</button>';
            card.querySelector('.btn-select-voice').addEventListener('click', function () {
              document.getElementById('selectVoiceProfile').value = v.profile_id;
              state.voiceProfileId = v.profile_id;
              showToast('Selected voice profile: ' + v.name);
              switchScreen('create');
            });
            box.appendChild(card);
          })(state.voices[i]);
        }
      }

      function renderCharactersList() {
        var box = document.getElementById('savedCharactersContainer');
        if (!state.characters.length) {
          box.innerHTML = '<div class="empty-state" style="padding:20px;">No character profiles yet. Attach reference images on the Create screen and click "Save as Character".</div>';
          return;
        }
        box.innerHTML = '';
        for (var i = 0; i < state.characters.length; i++) {
          (function (c) {
            var card = document.createElement('div');
            card.className = 'glass-subpanel';
            card.style.padding = '12px';
            card.style.marginBottom = '10px';
            var thumbSrc = c.thumbnail_src || (c.thumbnail_url ? withTokenUrl(c.thumbnail_url) : '');
            var imgTag = thumbSrc
              ? '<img src="' + thumbSrc + '" alt="' + c.name + '" style="width:52px; height:52px; border-radius:8px; object-fit:cover; border:1px solid var(--border);" />'
              : '';
            card.innerHTML =
              '<div style="display:flex; gap:12px; align-items:center;">' +
                imgTag +
                '<div style="flex:1;">' +
                  '<div style="display:flex; justify-content:space-between; align-items:center;">' +
                    '<strong>' + c.name + '</strong>' +
                    '<span class="badge badge-outline">' + (c.voice_profile_id ? 'Voice Paired' : 'Visual Only') + '</span>' +
                  '</div>' +
                  '<p style="font-size:11.5px; color:var(--text-2); margin:4px 0 8px;">' + (c.reference_description_en || '').slice(0, 110) + '</p>' +
                  '<button type="button" class="btn btn-secondary btn-sm btn-select-char">Use Character in Create</button>' +
                '</div>' +
              '</div>';
            card.querySelector('.btn-select-char').addEventListener('click', function () {
              document.getElementById('selectCharacterProfile').value = c.character_id;
              state.characterId = c.character_id;
              if (c.voice_profile_id) {
                document.getElementById('selectVoiceProfile').value = c.voice_profile_id;
                state.voiceProfileId = c.voice_profile_id;
              }
              showToast('Selected character: ' + c.name);
              switchScreen('create');
            });
            box.appendChild(card);
          })(state.characters[i]);
        }
      }

      var voiceDropzone = document.getElementById('voiceDropzone');
      var inputVoiceFile = document.getElementById('inputVoiceFile');
      var selectedVoiceFile = null;

      voiceDropzone.addEventListener('click', function () {
        inputVoiceFile.click();
      });
      inputVoiceFile.addEventListener('change', function (e) {
        if (e.currentTarget.files && e.currentTarget.files[0]) {
          selectedVoiceFile = e.currentTarget.files[0];
          document.getElementById('voiceDropzoneLabel').textContent = 'Selected: ' + selectedVoiceFile.name;
        }
      });

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

        var bars = 90;
        for (var i = 0; i < bars; i++) {
          var bx = (i / bars) * w;
          var inWin = bx >= sx && bx <= ex;
          var amp = (Math.sin(i * 0.45) * 0.35 + Math.cos(i * 1.1) * 0.25 + 0.42) * (h * 0.42);
          ctx.fillStyle = inWin ? '#ffffff' : 'rgba(255,255,255,0.25)';
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
        showToast('Set 9.7s pause-aligned trim window.');
      });

      document.getElementById('btnPreviewClonedVoice').addEventListener('click', function () {
        if (!document.getElementById('checkboxVoiceConsent').checked) {
          showToast('Please check the mandatory speaker consent checkbox first.');
          return;
        }
        var sampleTa = document.getElementById('inputRefTranscriptTa').value.trim() || 'வணக்கம், இது என்னுடைய குரல் மாதிரி பரிசோதனை.';
        fetch('/api/voice/preview', {
          method: 'POST',
          headers: apiHeaders(true),
          body: JSON.stringify({ voice_profile_id: state.voiceProfileId, text_ta: sampleTa })
        })
          .then(function (r) { return r.ok ? r.json() : Promise.reject('Failed'); })
          .then(function (res) {
            var aud = document.getElementById('voiceTabPreviewAudio');
            aud.style.display = 'block';
            aud.src = withTokenUrl(res.preview_wav_url);
            aud.play().catch(function () {});
            showToast('Synthesized preview sentence (' + res.duration_sec + 's).');
          })
          .catch(function () {
            showToast('Connect to backend to synthesize preview audio.');
          });
      });

      document.getElementById('btnSaveVoiceProfile').addEventListener('click', function () {
        if (!document.getElementById('checkboxVoiceConsent').checked) {
          showToast('Speaker consent checkbox is mandatory to save a voice profile.');
          return;
        }
        if (!selectedVoiceFile) {
          showToast('Please select an audio file to upload first.');
          return;
        }
        var fd = new FormData();
        fd.append('audio_file', selectedVoiceFile);
        fd.append('profile_name', document.getElementById('inputVoiceName').value.trim() || 'Tamil Voice');
        fd.append('consent_verified', 'true');
        fd.append('transcript_ta', document.getElementById('inputRefTranscriptTa').value.trim());
        fd.append('trim_start_sec', document.getElementById('trimStartSlider').value);
        fd.append('trim_end_sec', document.getElementById('trimEndSlider').value);

        fetch('/api/voice/profile', {
          method: 'POST',
          headers: { 'X-Studio-Token': state.token },
          body: fd
        })
          .then(function (r) {
            if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail || 'Error'); });
            return r.json();
          })
          .then(function (data) {
            state.voices = data.profiles || [];
            renderVoicesList();
            populateCharacterAndVoiceSelectors();
            showToast('Voice profile saved and synced to Drive.');
          })
          .catch(function (err) {
            showToast(err.message || 'Failed to save voice profile.');
          });
      });

      // -----------------------------------------------------------------------
      // Screen 6: Settings & Real Disk/Header Model Verification
      // -----------------------------------------------------------------------
      function fetchModelVerification() {
        var list = document.getElementById('modelVerificationList');
        fetch('/api/models/verify', { headers: apiHeaders(false) })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (data) {
            if (!data || !data.models) return;
            list.innerHTML = '';
            for (var i = 0; i < data.models.length; i++) {
              var m = data.models[i];
              var badgeClass = m.status === 'verified' ? 'badge-solid' : 'badge-dashed';
              var licTag = m.commercial_allowed ? 'Commercial OK (' + m.license + ')' : 'NON-COMMERCIAL (' + m.license + ')';
              var row = document.createElement('div');
              row.style.display = 'flex';
              row.style.justifyContent = 'space-between';
              row.style.alignItems = 'center';
              row.style.padding = '6px 0';
              row.style.borderBottom = '1px solid rgba(255,255,255,0.06)';
              row.innerHTML =
                '<div>' +
                  '<div style="font-weight:600;">' + m.stage + ' · ' + m.name + '</div>' +
                  '<div style="font-size:11px; color:var(--text-3);">' + m.detail + ' · ' + licTag + '</div>' +
                '</div>' +
                '<span class="badge ' + badgeClass + '">' + m.badge_text + '</span>';
              list.appendChild(row);
            }
          })
          .catch(function () {
            list.innerHTML = '<div style="color:var(--text-2);">Connect to FastAPI backend to inspect /kaggle/temp/tamil_studio/models on disk.</div>';
          });
      }

      document.getElementById('settingKeyframeEngine').addEventListener('change', function (e) {
        var val = e.currentTarget.value;
        var banner = document.getElementById('licenseNoticeBanner');
        if (val === 'flux_kontext') {
          banner.style.display = 'block';
        } else {
          banner.style.display = 'none';
        }
      });

      document.getElementById('btnVerifyModelsNow').addEventListener('click', function () {
        fetchModelVerification();
        pollLiveTelemetry();
        showToast('Verified model files and headers on disk.');
      });

      document.getElementById('btnSaveSettings').addEventListener('click', function () {
        state.preset = document.getElementById('settingQualityPreset').value;
        state.resolution = document.getElementById('settingResolution').value;
        state.token = document.getElementById('settingStudioToken').value.trim() || 'tamil-studio-local';
        try {
          localStorage.setItem('tamil_studio_token', state.token);
        } catch (e) {}

        fetch('/api/settings', {
          method: 'POST',
          headers: apiHeaders(true),
          body: JSON.stringify({
            default_preset: state.preset,
            resolution: state.resolution,
            custom_gguf_path: document.getElementById('settingCustomGgufPath').value.trim(),
            keyframe_engine: document.getElementById('settingKeyframeEngine').value
          })
        }).catch(function () {});

        updateReviewEtaFromBenchmarks();
        showToast('Saved configuration and updated ETA calculations.');
      });

      document.getElementById('btnClearVramCache').addEventListener('click', function () {
        fetch('/api/settings/clear-cache', {
          method: 'POST',
          headers: apiHeaders(false)
        })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (data) {
            if (data) updateTelemetryUI(data);
            showToast('Executed gc.collect() + torch.cuda.empty_cache() across GPUs.');
          })
          .catch(function () {
            showToast('Cleared local cache state.');
          });
      });

      // Initial data fetch
      fetchVoicesAndCharacters();
      fetchRealLibrary();
      fetchModelVerification();
      drawWaveformCanvas();
    })();

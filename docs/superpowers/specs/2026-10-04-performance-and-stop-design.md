# Performance and stop experience design

## Goal and scope

Make FlowLens feel responsive at launch, during local transcription, and after
Stop, without weakening model integrity, audio persistence, or the fully local
MVP. This is a design for three independently verifiable changes to existing
flows, not a claim that the designated PC meets its latency targets today.

## Observed baseline

- The startup preflight hashes the 1,512,927,867-byte ASR model and the
  2,497,279,008-byte discussion model in sequence. On the designated PC, one
  read-only PowerShell SHA-256 pass took 1.94 and 2.98 seconds respectively.
- Session start invokes another full readiness check. The discussion worker
  verifies its model again before constructing llama.cpp. Those checks protect
  integrity but contribute to startup and start-session delay.
- The ASR engine schedules partial decoding every 500 ms, uses a configured
  600 ms silence boundary by default, and performs a higher-quality final
  decode. A public 48-clip benchmark measured median final-decode time of
  0.376 seconds with beam 5, but it did not measure live two-source latency.
- After Stop, Audio drain, ASR finalization, final Discussion analysis, and
  Writer finalization run in acknowledgement order. The live UI currently
  presents only a generic `Finalizing` label and offers a slow-finalization
  choice after 30 seconds. There is no designated-PC stop-to-completion timing.

## Launch responsiveness

Run the two complete model checksum calculations concurrently during preflight.
Each file must still be fully read and matched against its exact manifest hash;
missing, changed, linked, or invalid files still block Start. Keep recovery,
device discovery, and integrity I/O off the GUI thread. Measure shell-visible
and controls-ready times separately because parallel hashing may contend with
the disk and is not guaranteed to reduce wall time on every machine. Do not
introduce a persistent trust cache or skip later session-start checks in this
change.

## Transcription responsiveness

First collect content-free timings at capture-to-ASR, ASR decode, and
ASR-to-UI boundaries, including per-source backlog and p95 partial/commit
latency. Use the existing local acceptance-report pathway or another opt-in
local diagnostic; never log audio or transcript content. Investigate the
measured dominant stage before changing cadence, VAD, decoding, or scheduling.
Preserve the beam-5 final quality path and the current noise-rejection gates
unless a before/after benchmark demonstrates a justified trade-off. No cloud
processing, automatic model download, or physical speaker playback is added.

## Stop experience

After Stop confirmation, make it immediately clear that capture has stopped
and the remaining work is finalization. Show the current acknowledged stage in
the live page: draining captured audio, finishing transcription, updating the
discussion summary, or saving the session. Avoid a fake percentage or ETA.
Continue showing elapsed wait time and the most recent successful save time.
The 30-second choice stays explicit and non-destructive by default. Explain
that Force close can leave the final transcript or summary incomplete; never
select it automatically. Closing the window during finalization should direct
attention to the same visible progress and choices. The completion page appears
only after the Writer's durable completion acknowledgement.

## Verification and boundaries

Use test-first changes for state transitions, UI visibility, and checksum
behavior. Test concurrent integrity checks against missing and mismatched
files, including deterministic ordering of returned results. Exercise every
finalization stage and the 30-second choice with fake workers and offscreen Qt.
Run the full automated suite, static checks, and rebuild `dist/FlowLens`.
Compare startup timings on the designated PC without recording or playing
audio. Report transcription and stop latency as unverified until a permitted
real session supplies measurements; synthetic timings do not establish live
acceptance.

## Exclusions

This design does not change the required finalization order, bypass checksums,
auto-force-close, alter stored transcript records, change local-only privacy,
or claim that an offscreen UI test validates Windows native chrome.

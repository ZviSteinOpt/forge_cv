const vscode = require('vscode');
const fs = require('fs');
const path = require('path');

let lastVarName = null;
let matPanel = null;
let matImages = new Map(); // name -> mat object
let trackedVarNames = [];  // ordered list of variable names

// ---- Debug session helpers ----

async function evaluate(session, expr) {
    try {
        const frameId = await getTopFrameId(session);
        const resp = await session.customRequest('evaluate', {
            expression: expr,
            frameId: frameId,
            context: 'watch'
        });
        return resp.result;
    } catch (e) {
        return null;
    }
}

async function getTopFrameId(session) {
    const threads = await session.customRequest('threads');
    let threadId = threads.threads[0].id;
    for (const t of threads.threads) {
        try {
            const stack = await session.customRequest('stackTrace', {
                threadId: t.id,
                startFrame: 0,
                levels: 1
            });
            if (stack.stackFrames && stack.stackFrames.length > 0) {
                const frame = stack.stackFrames[0];
                if (frame.source && frame.source.path) {
                    return frame.id;
                }
            }
        } catch (e) { /* thread may not have frames */ }
    }
    const stack = await session.customRequest('stackTrace', {
        threadId: threadId,
        startFrame: 0,
        levels: 1
    });
    return stack.stackFrames[0].id;
}

async function readMatFromDebugger(session, varName) {
    const rowsStr = await evaluate(session, `${varName}.rows`);
    const colsStr = await evaluate(session, `${varName}.cols`);
    const channelsStr = await evaluate(session, `${varName}.channels`);
    const typeStr = await evaluate(session, `${varName}.scalar_type`);
    const lenStr = await evaluate(session, `${varName}.data_len`);
    const ptrStr = await evaluate(session, `${varName}.data_ptr`);

    const diag = `rows=${JSON.stringify(rowsStr)}, cols=${JSON.stringify(colsStr)}, channels=${JSON.stringify(channelsStr)}, type=${JSON.stringify(typeStr)}, len=${JSON.stringify(lenStr)}, ptr=${JSON.stringify(ptrStr)}`;

    if (!rowsStr || !colsStr || !ptrStr || !lenStr) {
        throw new Error(`Could not read '${varName}'. Debug info:\n${diag}`);
    }

    const rows = parseInt(rowsStr);
    const cols = parseInt(colsStr);
    const channels = parseInt(channelsStr) || 1;
    const dataLen = parseInt(lenStr);
    const ptrAddr = parseInt(ptrStr);

    let typeId = 2;
    const ts = (typeStr || '').toLowerCase();
    if (ts.includes('uint8'))   typeId = 0;
    if (ts.includes('int32'))   typeId = 1;
    if (ts.includes('float32')) typeId = 2;
    if (ts.includes('float64')) typeId = 3;

    const typeInfo = [
        { name: 'uint8',   size: 1 },
        { name: 'int32',   size: 4 },
        { name: 'float32', size: 4 },
        { name: 'float64', size: 8 },
    ][typeId];

    const memResp = await session.customRequest('readMemory', {
        memoryReference: `0x${ptrAddr.toString(16)}`,
        offset: 0,
        count: dataLen
    });

    const rawBytes = Buffer.from(memResp.data, 'base64');

    return {
        name: varName, rows, cols, channels,
        typeName: typeInfo.name, typeId,
        mode: 'canvas',
        rawBase64: rawBytes.toString('base64'),
    };
}

// ---- WebviewPanel manager ----

function getOrCreatePanel() {
    if (matPanel && !matPanel.disposed) {
        matPanel.reveal(vscode.ViewColumn.Beside, true);
        return matPanel;
    }
    matPanel = vscode.window.createWebviewPanel(
        'gmat.matView',
        'Mat Viewer',
        { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
        { enableScripts: true, retainContextWhenHidden: true }
    );
    matPanel.onDidDispose(() => { matPanel = null; });
    return matPanel;
}

function updatePanel(mat) {
    const panel = getOrCreatePanel();
    matImages.set(mat.name, mat);
    if (!trackedVarNames.includes(mat.name)) {
        trackedVarNames.push(mat.name);
    }
    const allImages = trackedVarNames
        .filter(n => matImages.get(n))
        .map(n => matImages.get(n));
    panel.webview.html = getMultiCanvasHtml(allImages);
}

// ---- Commands ----

function activate(context) {
    // Command: view a Mat variable from the debug session
    context.subscriptions.push(
        vscode.commands.registerCommand('gmat-viewer.viewVariable', async (variableArg) => {
            const session = vscode.debug.activeDebugSession;
            if (!session) {
                vscode.window.showErrorMessage('No active debug session');
                return;
            }

            let varName;
            if (variableArg && variableArg.variable && variableArg.variable.name) {
                varName = variableArg.variable.name;
            } else if (variableArg && variableArg.container && variableArg.variable) {
                varName = variableArg.variable.name;
            } else {
                varName = await vscode.window.showInputBox({
                    prompt: 'Variable name (e.g. "mat" or "my_struct.matrix")',
                    value: lastVarName || 'mat',
                });
            }

            if (!varName) return;
            lastVarName = varName;

            try {
                const mat = await readMatFromDebugger(session, varName);
                updatePanel(mat);
            } catch (e) {
                vscode.window.showErrorMessage(`GMAT: ${e.message}`);
            }
        })
    );

    // Auto-refresh on debugger stop
    context.subscriptions.push(
        vscode.debug.registerDebugAdapterTrackerFactory('lldb', {
            createDebugAdapterTracker(session) {
                return {
                    onDidSendMessage(message) {
                        if (message.type === 'event' && message.event === 'stopped') {
                            if (trackedVarNames.length > 0 && matPanel && !matPanel.disposed) {
                                setTimeout(async () => {
                                    for (const varName of trackedVarNames) {
                                        try {
                                            const mat = await readMatFromDebugger(session, varName);
                                            matImages.set(varName, mat);
                                        } catch (e) { /* variable out of scope */ }
                                    }
                                    // Re-render with latest data
                                    const allImages = trackedVarNames
                                        .filter(n => matImages.get(n))
                                        .map(n => matImages.get(n));
                                    if (allImages.length > 0) {
                                        matPanel.webview.html = getMultiCanvasHtml(allImages);
                                    }
                                }, 200);
                            }
                        }
                    }
                };
            }
        })
    );

    // File-based .gmat viewer
    const provider = new GmatFileEditorProvider(context);
    context.subscriptions.push(
        vscode.window.registerCustomEditorProvider('gmat-viewer.matrixView', provider, {
            webviewOptions: { retainContextWhenHidden: true },
            supportsMultipleEditorsPerDocument: false,
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('gmat-viewer.open', async () => {
            const uris = await vscode.window.showOpenDialog({
                filters: { 'GMAT files': ['gmat'] },
                canSelectMany: false,
            });
            if (uris && uris[0]) {
                await vscode.commands.executeCommand('vscode.openWith', uris[0], 'gmat-viewer.matrixView');
            }
        })
    );
}

// ---- .gmat file editor ----

class GmatFileEditorProvider {
    constructor(context) { this.context = context; }

    resolveCustomEditor(document, webviewPanel) {
        webviewPanel.webview.options = { enableScripts: true };
        const loadAndRender = () => {
            try {
                const bytes = new Uint8Array(fs.readFileSync(document.uri.fsPath));
                const mat = parseGmatFile(bytes);
                webviewPanel.webview.html = getCanvasHtml(mat);
            } catch (e) {
                webviewPanel.webview.html = `<pre>Error: ${e.message}</pre>`;
            }
        };
        loadAndRender();
        const watcher = fs.watch(document.uri.fsPath, () => loadAndRender());
        webviewPanel.onDidDispose(() => watcher.close());
    }
}

function parseGmatFile(buffer) {
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const magic = String.fromCharCode(buffer[0], buffer[1], buffer[2], buffer[3]);
    if (magic !== 'GMAT') throw new Error('Invalid .gmat file');

    const rows = view.getUint32(4, true);
    const cols = view.getUint32(8, true);
    const channels = view.getUint32(12, true);
    const typeId = view.getUint32(16, true);
    const nameLen = view.getUint32(20, true);
    const name = new TextDecoder().decode(buffer.slice(24, 24 + nameLen));
    const dataBytes = buffer.slice(24 + nameLen);

    const typeInfo = [
        { name: 'uint8', size: 1 }, { name: 'int32', size: 4 },
        { name: 'float32', size: 4 }, { name: 'float64', size: 8 },
    ][typeId];

    return {
        name, rows, cols, channels,
        typeName: typeInfo.name, typeId,
        mode: 'canvas',
        rawBase64: Buffer.from(dataBytes).toString('base64'),
    };
}

// ===========================================================================
// ---- HTML generators ----
// ===========================================================================

function getEmptyHtml() {
    return /*html*/`<!DOCTYPE html><html><body style="
        font-family: 'Menlo', monospace; font-size: 12px;
        color: var(--vscode-descriptionForeground);
        display: flex; align-items: center; justify-content: center; height: 100vh;
    "><p>Stop at a breakpoint, then:<br>Cmd+Shift+P → GMAT: View Matrix Variable</p></body></html>`;
}

// ---- Canvas mode (images / large matrices) ----
// Uses a full-viewport canvas with manual rendering.
// At low zoom: smooth image. At high zoom: pixel grid + value labels.

function getCanvasHtml(mat) {
    return /*html*/`<!DOCTYPE html>
<html>
<head>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Menlo', 'Consolas', monospace;
        font-size: 11px;
        background: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
        overflow: hidden;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    .header {
        padding: 4px 8px;
        background: var(--vscode-editorWidget-background);
        border-bottom: 1px solid var(--vscode-editorWidget-border);
        display: flex;
        gap: 12px;
        align-items: center;
        flex-shrink: 0;
    }
    .header .name { font-weight: bold; }
    .header .dim { color: var(--vscode-descriptionForeground); }
    .canvas-wrap {
        flex: 1;
        overflow: hidden;
        position: relative;
        cursor: crosshair;
    }
    canvas {
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
    }
    .tooltip {
        position: fixed;
        background: var(--vscode-editorHoverWidget-background);
        border: 1px solid var(--vscode-editorHoverWidget-border);
        padding: 3px 6px;
        border-radius: 3px;
        pointer-events: none;
        display: none;
        z-index: 100;
        font-size: 11px;
        white-space: pre;
    }
</style>
</head>
<body>
<div class="header">
    <span class="name">${mat.name}</span>
    <span class="dim">${mat.rows} × ${mat.cols} × ${mat.channels}ch</span>
    <span class="dim">${mat.typeName}</span>
    <span class="dim" id="zoom-label">Fit</span>
    <label class="dim" style="cursor:pointer;"><input type="checkbox" id="norm-cb"> Norm</label>
    <span class="dim" id="pixel-info"></span>
</div>
<div class="canvas-wrap" id="wrap">
    <canvas id="cv"></canvas>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
    const rows = ${mat.rows};
    const cols = ${mat.cols};
    const channels = ${mat.channels};
    const typeId = ${mat.typeId};
    const rawBase64 = "${mat.rawBase64}";

    // Decode base64 → Uint8Array
    const raw = Uint8Array.from(atob(rawBase64), c => c.charCodeAt(0));
    const bytesPerElem = [1, 4, 4, 8][typeId];

    // Read all raw float/int values for normalization
    function readRawValue(byteOffset) {
        const dv = new DataView(raw.buffer, raw.byteOffset + byteOffset, bytesPerElem);
        switch (typeId) {
            case 0: return raw[byteOffset];
            case 1: return dv.getInt32(0, true);
            case 2: return dv.getFloat32(0, true);
            case 3: return dv.getFloat64(0, true);
        }
        return 0;
    }

    // Compute min/max across all values (for normalize)
    let globalMin = Infinity, globalMax = -Infinity;
    if (typeId >= 1) {
        for (let i = 0; i < rows * cols * channels; i++) {
            const v = readRawValue(i * bytesPerElem);
            if (v < globalMin) globalMin = v;
            if (v > globalMax) globalMax = v;
        }
    }

    // Build offscreen image + pixelRGBA
    const offscreen = document.createElement('canvas');
    offscreen.width = cols;
    offscreen.height = rows;
    const offCtx = offscreen.getContext('2d');
    const pixelRGBA = new Uint8Array(rows * cols * 4);

    function rebuildImage(normalize) {
        const imgData = offCtx.createImageData(cols, rows);
        const d = imgData.data;
        const range = globalMax - globalMin;

        for (let i = 0; i < rows * cols; i++) {
            const dstOff = i * 4;
            let r = 0, g = 0, b = 0, a = 255;
            if (typeId === 0 && channels === 3) {
                const srcOff = i * 3;
                r = raw[srcOff]; g = raw[srcOff + 1]; b = raw[srcOff + 2];
            } else if (typeId === 0 && channels === 4) {
                const srcOff = i * 4;
                r = raw[srcOff]; g = raw[srcOff + 1]; b = raw[srcOff + 2]; a = raw[srcOff + 3];
            } else if (typeId === 0 && channels === 1) {
                r = g = b = raw[i];
            } else if (channels >= 3) {
                const srcOff = i * channels * bytesPerElem;
                const rv = readRawValue(srcOff);
                const gv = readRawValue(srcOff + bytesPerElem);
                const bv = readRawValue(srcOff + 2 * bytesPerElem);
                if (normalize && range > 0) {
                    r = Math.round(((rv - globalMin) / range) * 255);
                    g = Math.round(((gv - globalMin) / range) * 255);
                    b = Math.round(((bv - globalMin) / range) * 255);
                } else {
                    r = Math.max(0, Math.min(255, Math.round(rv)));
                    g = Math.max(0, Math.min(255, Math.round(gv)));
                    b = Math.max(0, Math.min(255, Math.round(bv)));
                }
            } else {
                const srcOff = i * channels * bytesPerElem;
                let v = readRawValue(srcOff);
                if (normalize && range > 0) {
                    v = ((v - globalMin) / range) * 255;
                }
                r = g = b = Math.max(0, Math.min(255, Math.round(v)));
            }
            d[dstOff] = r; d[dstOff+1] = g; d[dstOff+2] = b; d[dstOff+3] = a;
            pixelRGBA[dstOff] = r; pixelRGBA[dstOff+1] = g; pixelRGBA[dstOff+2] = b; pixelRGBA[dstOff+3] = a;
        }
        offCtx.putImageData(imgData, 0, 0);
    }
    rebuildImage(false);

    document.getElementById('norm-cb').addEventListener('change', function() {
        rebuildImage(this.checked);
        render();
    });

    // Helper: read raw channel values for a pixel
    function getPixelValues(py, px) {
        const idx = (py * cols + px) * channels;
        const srcOff = idx * bytesPerElem;
        const vals = [];
        for (let ch = 0; ch < channels; ch++) {
            const off = srcOff + ch * bytesPerElem;
            const dv = new DataView(raw.buffer, raw.byteOffset + off, bytesPerElem);
            switch (typeId) {
                case 0: vals.push(raw[off]); break;
                case 1: vals.push(dv.getInt32(0, true)); break;
                case 2: vals.push(dv.getFloat32(0, true)); break;
                case 3: vals.push(dv.getFloat64(0, true)); break;
            }
        }
        return vals;
    }

    // ---- Main canvas (fills viewport) ----
    const cv = document.getElementById('cv');
    const ctx = cv.getContext('2d');
    const wrap = document.getElementById('wrap');
    const zoomLabel = document.getElementById('zoom-label');
    const pixelInfo = document.getElementById('pixel-info');
    const tooltip = document.getElementById('tooltip');

    let scale = 1;
    let panX = 0, panY = 0;
    let dragging = false;
    let dragStartX, dragStartY, panStartX, panStartY;

    const GRID_THRESHOLD = 8;   // px per pixel to start showing grid lines
    const TEXT_THRESHOLD = 40;  // px per pixel to start showing values

    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        cv.width = wrap.clientWidth * dpr;
        cv.height = wrap.clientHeight * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function fitToView() {
        const ww = wrap.clientWidth;
        const wh = wrap.clientHeight;
        scale = Math.min(ww / cols, wh / rows);
        panX = (ww - cols * scale) / 2;
        panY = (wh - rows * scale) / 2;
        render();
    }

    function render() {
        const ww = wrap.clientWidth;
        const wh = wrap.clientHeight;
        ctx.clearRect(0, 0, ww, wh);

        zoomLabel.textContent = (scale * 100).toFixed(0) + '%';

        if (scale < GRID_THRESHOLD) {
            // Low zoom: draw the offscreen image scaled smoothly
            ctx.imageSmoothingEnabled = (scale < 1);
            ctx.drawImage(offscreen, panX, panY, cols * scale, rows * scale);
        } else {
            // High zoom: draw individual pixel rects
            ctx.imageSmoothingEnabled = false;

            // Calculate visible pixel range
            const startCol = Math.max(0, Math.floor(-panX / scale));
            const endCol = Math.min(cols, Math.ceil((ww - panX) / scale));
            const startRow = Math.max(0, Math.floor(-panY / scale));
            const endRow = Math.min(rows, Math.ceil((wh - panY) / scale));

            // Draw pixel rectangles
            for (let py = startRow; py < endRow; py++) {
                for (let px = startCol; px < endCol; px++) {
                    const off = (py * cols + px) * 4;
                    const r = pixelRGBA[off], g = pixelRGBA[off+1], b = pixelRGBA[off+2], a = pixelRGBA[off+3];
                    ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + (a / 255) + ')';
                    ctx.fillRect(
                        panX + px * scale,
                        panY + py * scale,
                        scale,
                        scale
                    );
                }
            }

            // Draw grid lines
            ctx.strokeStyle = 'rgba(128, 128, 128, 0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let px = startCol; px <= endCol; px++) {
                const x = Math.round(panX + px * scale) + 0.5;
                ctx.moveTo(x, panY + startRow * scale);
                ctx.lineTo(x, panY + endRow * scale);
            }
            for (let py = startRow; py <= endRow; py++) {
                const y = Math.round(panY + py * scale) + 0.5;
                ctx.moveTo(panX + startCol * scale, y);
                ctx.lineTo(panX + endCol * scale, y);
            }
            ctx.stroke();

            // Draw pixel values when zoomed in enough
            if (scale >= TEXT_THRESHOLD) {
                const fontSize = Math.max(8, Math.min(14, scale / (channels >= 3 ? 5 : 4)));
                ctx.font = fontSize + 'px Menlo, Consolas, monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                for (let py = startRow; py < endRow; py++) {
                    for (let px = startCol; px < endCol; px++) {
                        const off = (py * cols + px) * 4;
                        const r = pixelRGBA[off], g = pixelRGBA[off+1], b = pixelRGBA[off+2];
                        // Choose text color for contrast
                        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
                        ctx.fillStyle = lum > 128 ? '#000' : '#fff';

                        const cx = panX + (px + 0.5) * scale;
                        const cy = panY + (py + 0.5) * scale;

                        const vals = getPixelValues(py, px);
                        if (channels === 1) {
                            // Single value centered
                            const txt = typeId >= 2 ? vals[0].toFixed(4) : vals[0].toString();
                            ctx.fillText(txt, cx, cy);
                        } else {
                            // Stack channel values vertically
                            const lineH = fontSize + 2;
                            const totalH = channels * lineH;
                            const topY = cy - totalH / 2 + lineH / 2;
                            for (let ch = 0; ch < channels; ch++) {
                                const txt = typeId >= 2 ? vals[ch].toFixed(4) : vals[ch].toString();
                                ctx.fillText(txt, cx, topY + ch * lineH);
                            }
                        }
                    }
                }
            }
        }
    }

    // Zoom on wheel (centered on cursor)
    wrap.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = wrap.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const oldScale = scale;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        scale = Math.max(0.01, Math.min(200, scale * factor));

        // Zoom toward cursor
        panX = mx - (mx - panX) * (scale / oldScale);
        panY = my - (my - panY) * (scale / oldScale);
        render();
    }, { passive: false });

    // Pan on drag
    wrap.addEventListener('mousedown', (e) => {
        dragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        panStartX = panX;
        panStartY = panY;
        wrap.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
        if (dragging) {
            panX = panStartX + (e.clientX - dragStartX);
            panY = panStartY + (e.clientY - dragStartY);
            render();
        }

        // Show pixel info on hover
        const rect = wrap.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const px = Math.floor((mx - panX) / scale);
        const py = Math.floor((my - panY) / scale);

        if (px >= 0 && px < cols && py >= 0 && py < rows) {
            const vals = getPixelValues(py, px);
            const vstr = vals.map(v => typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : v).join(', ');
            pixelInfo.textContent = '[' + py + ', ' + px + '] = (' + vstr + ')';
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top = (e.clientY + 12) + 'px';
            tooltip.textContent = '[' + py + ', ' + px + '] = (' + vstr + ')';
        } else {
            pixelInfo.textContent = '';
            tooltip.style.display = 'none';
        }
    });
    window.addEventListener('mouseup', () => {
        dragging = false;
        wrap.style.cursor = 'crosshair';
    });

    // Initial setup
    resizeCanvas();
    fitToView();
    window.addEventListener('resize', () => { resizeCanvas(); render(); });
</script>
</body>
</html>`;
}

// ---- Multi-image canvas mode ----
// Shows one image at a time with a sidebar to switch.
// "Link View": when switching images, pan/zoom is preserved (for comparing same-size images).

function getMultiCanvasHtml(images) {
    const imagesJson = JSON.stringify(images.map(img => ({
        name: img.name, rows: img.rows, cols: img.cols,
        channels: img.channels, typeId: img.typeId,
        typeName: img.typeName, rawBase64: img.rawBase64,
    })));

    return /*html*/`<!DOCTYPE html>
<html>
<head>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: 'Menlo', 'Consolas', monospace;
        font-size: 11px;
        background: var(--vscode-editor-background);
        color: var(--vscode-editor-foreground);
        overflow: hidden;
        height: 100vh;
        display: flex;
    }
    .sidebar {
        width: 160px;
        min-width: 120px;
        background: var(--vscode-sideBar-background, var(--vscode-editorWidget-background));
        border-right: 1px solid var(--vscode-editorWidget-border);
        overflow-y: auto;
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
    }
    .section-title {
        padding: 6px 8px 2px;
        font-size: 10px;
        text-transform: uppercase;
        color: var(--vscode-descriptionForeground);
        letter-spacing: 0.5px;
    }
    .img-item {
        padding: 4px 8px;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        gap: 1px;
        border-left: 3px solid transparent;
    }
    .img-item:hover { background: var(--vscode-list-hoverBackground); }
    .img-item.selected {
        background: var(--vscode-list-activeSelectionBackground);
        color: var(--vscode-list-activeSelectionForeground);
        border-left-color: var(--vscode-focusBorder);
    }
    .img-item .img-name {
        font-weight: bold; font-size: 11px;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .img-item .img-dim {
        font-size: 10px;
        color: var(--vscode-descriptionForeground);
    }
    .img-item.selected .img-dim {
        color: var(--vscode-list-activeSelectionForeground);
        opacity: 0.7;
    }
    .link-section {
        padding: 6px 8px;
        border-top: 1px solid var(--vscode-editorWidget-border);
        margin-top: auto;
    }
    .link-toggle {
        display: flex; align-items: center; gap: 4px;
        cursor: pointer; font-size: 10px;
    }
    .link-toggle input { cursor: pointer; }
    .link-hint {
        font-size: 9px; color: var(--vscode-descriptionForeground);
        margin-top: 2px;
    }
    .right-panel {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    .header {
        padding: 4px 8px;
        background: var(--vscode-editorWidget-background);
        border-bottom: 1px solid var(--vscode-editorWidget-border);
        display: flex;
        gap: 12px;
        align-items: center;
        flex-shrink: 0;
    }
    .header .name { font-weight: bold; }
    .header .dim { color: var(--vscode-descriptionForeground); }
    .canvas-wrap {
        flex: 1;
        overflow: hidden;
        position: relative;
        cursor: crosshair;
    }
    canvas {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
    }
    .tooltip {
        position: fixed;
        background: var(--vscode-editorHoverWidget-background);
        border: 1px solid var(--vscode-editorHoverWidget-border);
        padding: 3px 6px; border-radius: 3px;
        pointer-events: none; display: none; z-index: 100;
        font-size: 11px; white-space: pre;
    }
</style>
</head>
<body>
<div class="sidebar" id="sidebar"></div>
<div class="right-panel">
    <div class="header">
        <span class="name" id="hdr-name"></span>
        <span class="dim" id="hdr-dim"></span>
        <span class="dim" id="zoom-label">Fit</span>
        <label class="dim" style="cursor:pointer;"><input type="checkbox" id="norm-cb"> Norm</label>
        <span class="dim" id="pixel-info"></span>
    </div>
    <div class="canvas-wrap" id="wrap">
        <canvas id="cv"></canvas>
    </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
    const allImages = ${imagesJson};
    const GRID_THRESHOLD = 8;
    const TEXT_THRESHOLD = 40;

    // ---- Decode all images upfront ----
    function readRawValueMulti(raw, byteOffset, bytesPerElem, typeId) {
        const dv = new DataView(raw.buffer, raw.byteOffset + byteOffset, bytesPerElem);
        switch (typeId) {
            case 0: return raw[byteOffset];
            case 1: return dv.getInt32(0, true);
            case 2: return dv.getFloat32(0, true);
            case 3: return dv.getFloat64(0, true);
        }
        return 0;
    }

    const decoded = allImages.map(img => {
        const raw = Uint8Array.from(atob(img.rawBase64), c => c.charCodeAt(0));
        const bytesPerElem = [1, 4, 4, 8][img.typeId];

        // Compute min/max for normalization
        let gMin = Infinity, gMax = -Infinity;
        if (img.typeId >= 1) {
            for (let i = 0; i < img.rows * img.cols * img.channels; i++) {
                const v = readRawValueMulti(raw, i * bytesPerElem, bytesPerElem, img.typeId);
                if (v < gMin) gMin = v;
                if (v > gMax) gMax = v;
            }
        }

        const offscreen = document.createElement('canvas');
        offscreen.width = img.cols;
        offscreen.height = img.rows;
        const offCtx = offscreen.getContext('2d');
        const pixelRGBA = new Uint8Array(img.rows * img.cols * 4);

        function rebuildImage(normalize) {
            const imgData = offCtx.createImageData(img.cols, img.rows);
            const d = imgData.data;
            const range = gMax - gMin;

            for (let i = 0; i < img.rows * img.cols; i++) {
                const dstOff = i * 4;
                let r = 0, g = 0, b = 0, a = 255;
                if (img.typeId === 0 && img.channels === 3) {
                    const s = i * 3; r = raw[s]; g = raw[s+1]; b = raw[s+2];
                } else if (img.typeId === 0 && img.channels === 4) {
                    const s = i * 4; r = raw[s]; g = raw[s+1]; b = raw[s+2]; a = raw[s+3];
                } else if (img.typeId === 0 && img.channels === 1) {
                    r = g = b = raw[i];
                } else if (img.channels >= 3) {
                    const s = i * img.channels * bytesPerElem;
                    const rv = readRawValueMulti(raw, s, bytesPerElem, img.typeId);
                    const gv = readRawValueMulti(raw, s + bytesPerElem, bytesPerElem, img.typeId);
                    const bv = readRawValueMulti(raw, s + 2 * bytesPerElem, bytesPerElem, img.typeId);
                    if (normalize && range > 0) {
                        r = Math.round(((rv - gMin) / range) * 255);
                        g = Math.round(((gv - gMin) / range) * 255);
                        b = Math.round(((bv - gMin) / range) * 255);
                    } else {
                        r = Math.max(0, Math.min(255, Math.round(rv)));
                        g = Math.max(0, Math.min(255, Math.round(gv)));
                        b = Math.max(0, Math.min(255, Math.round(bv)));
                    }
                } else {
                    const s = i * img.channels * bytesPerElem;
                    let v = readRawValueMulti(raw, s, bytesPerElem, img.typeId);
                    if (normalize && range > 0) {
                        v = ((v - gMin) / range) * 255;
                    }
                    r = g = b = Math.max(0, Math.min(255, Math.round(v)));
                }
                d[dstOff]=r; d[dstOff+1]=g; d[dstOff+2]=b; d[dstOff+3]=a;
                pixelRGBA[dstOff]=r; pixelRGBA[dstOff+1]=g; pixelRGBA[dstOff+2]=b; pixelRGBA[dstOff+3]=a;
            }
            offCtx.putImageData(imgData, 0, 0);
        }
        rebuildImage(false);

        function getPixelValues(py, px) {
            const idx = (py * img.cols + px) * img.channels;
            const srcOff = idx * bytesPerElem;
            const vals = [];
            for (let ch = 0; ch < img.channels; ch++) {
                const off = srcOff + ch * bytesPerElem;
                vals.push(readRawValueMulti(raw, off, bytesPerElem, img.typeId));
            }
            return vals;
        }

        return { ...img, raw, bytesPerElem, offscreen, pixelRGBA, getPixelValues, rebuildImage };
    });

    // ---- State ----
    let selectedIndex = decoded.length - 1;
    let linkedView = false;
    let scale = 1, panX = 0, panY = 0;
    let dragging = false, dragStartX, dragStartY, panStartX, panStartY;

    const sidebar = document.getElementById('sidebar');
    const cv = document.getElementById('cv');
    const ctx = cv.getContext('2d');
    const wrap = document.getElementById('wrap');
    const zoomLabel = document.getElementById('zoom-label');
    const pixelInfo = document.getElementById('pixel-info');
    const tooltip = document.getElementById('tooltip');
    const hdrName = document.getElementById('hdr-name');
    const hdrDim = document.getElementById('hdr-dim');

    function curImg() { return decoded[selectedIndex]; }

    // ---- Sidebar ----
    function buildSidebar() {
        let html = '<div class="section-title">Images</div>';
        decoded.forEach((img, i) => {
            const sel = i === selectedIndex ? ' selected' : '';
            html += '<div class="img-item' + sel + '" data-idx="' + i + '">'
                + '<span class="img-name">' + img.name + '</span>'
                + '<span class="img-dim">' + img.rows + '×' + img.cols + '×' + img.channels + 'ch</span>'
                + '</div>';
        });

        if (decoded.length >= 2) {
            html += '<div class="link-section">'
                + '<label class="link-toggle"><input type="checkbox" id="link-chk"'
                + (linkedView ? ' checked' : '') + '> Link View</label>'
                + '<div class="link-hint">' + (linkedView ? 'Pan/zoom preserved when switching' : 'Keep position when switching') + '</div>'
                + '</div>';
        }

        sidebar.innerHTML = html;

        sidebar.querySelectorAll('.img-item').forEach(el => {
            el.addEventListener('click', () => {
                const idx = parseInt(el.dataset.idx);
                if (idx === selectedIndex) return;
                selectedIndex = idx;
                buildSidebar();
                updateHeader();
                if (!linkedView) {
                    fitToView();
                } else {
                    render();
                }
            });
        });

        const linkChk = document.getElementById('link-chk');
        if (linkChk) {
            linkChk.addEventListener('change', () => {
                linkedView = linkChk.checked;
                buildSidebar();
            });
        }
    }

    document.getElementById('norm-cb').addEventListener('change', function() {
        const normalize = this.checked;
        decoded.forEach(d => d.rebuildImage(normalize));
        render();
    });

    function updateHeader() {
        const img = curImg();
        hdrName.textContent = img.name;
        hdrDim.textContent = img.rows + ' × ' + img.cols + ' × ' + img.channels + 'ch  ' + img.typeName;
    }

    // ---- Canvas ----
    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        cv.width = wrap.clientWidth * dpr;
        cv.height = wrap.clientHeight * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function fitToView() {
        const img = curImg();
        const ww = wrap.clientWidth;
        const wh = wrap.clientHeight;
        scale = Math.min(ww / img.cols, wh / img.rows);
        panX = (ww - img.cols * scale) / 2;
        panY = (wh - img.rows * scale) / 2;
        render();
    }

    function render() {
        const img = curImg();
        const ww = wrap.clientWidth;
        const wh = wrap.clientHeight;
        ctx.clearRect(0, 0, ww, wh);
        zoomLabel.textContent = (scale * 100).toFixed(0) + '%';

        if (scale < GRID_THRESHOLD) {
            ctx.imageSmoothingEnabled = (scale < 1);
            ctx.drawImage(img.offscreen, panX, panY, img.cols * scale, img.rows * scale);
        } else {
            ctx.imageSmoothingEnabled = false;
            const startCol = Math.max(0, Math.floor(-panX / scale));
            const endCol = Math.min(img.cols, Math.ceil((ww - panX) / scale));
            const startRow = Math.max(0, Math.floor(-panY / scale));
            const endRow = Math.min(img.rows, Math.ceil((wh - panY) / scale));

            for (let r = startRow; r < endRow; r++) {
                for (let c = startCol; c < endCol; c++) {
                    const off = (r * img.cols + c) * 4;
                    const rv = img.pixelRGBA[off], gv = img.pixelRGBA[off+1], bv = img.pixelRGBA[off+2], av = img.pixelRGBA[off+3];
                    ctx.fillStyle = 'rgba(' + rv + ',' + gv + ',' + bv + ',' + (av/255) + ')';
                    ctx.fillRect(panX + c * scale, panY + r * scale, scale, scale);
                }
            }

            ctx.strokeStyle = 'rgba(128,128,128,0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let c = startCol; c <= endCol; c++) {
                const x = Math.round(panX + c * scale) + 0.5;
                ctx.moveTo(x, panY + startRow * scale);
                ctx.lineTo(x, panY + endRow * scale);
            }
            for (let r = startRow; r <= endRow; r++) {
                const y = Math.round(panY + r * scale) + 0.5;
                ctx.moveTo(panX + startCol * scale, y);
                ctx.lineTo(panX + endCol * scale, y);
            }
            ctx.stroke();

            if (scale >= TEXT_THRESHOLD) {
                const fontSize = Math.max(8, Math.min(14, scale / (img.channels >= 3 ? 5 : 4)));
                ctx.font = fontSize + 'px Menlo, Consolas, monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                for (let r = startRow; r < endRow; r++) {
                    for (let c = startCol; c < endCol; c++) {
                        const off = (r * img.cols + c) * 4;
                        const rv = img.pixelRGBA[off], gv = img.pixelRGBA[off+1], bv = img.pixelRGBA[off+2];
                        const lum = 0.299*rv + 0.587*gv + 0.114*bv;
                        ctx.fillStyle = lum > 128 ? '#000' : '#fff';
                        const cx = panX + (c + 0.5) * scale;
                        const cy = panY + (r + 0.5) * scale;
                        const vals = img.getPixelValues(r, c);
                        if (img.channels === 1) {
                            ctx.fillText(img.typeId >= 2 ? vals[0].toFixed(4) : vals[0].toString(), cx, cy);
                        } else {
                            const lineH = fontSize + 2;
                            const totalH = img.channels * lineH;
                            const topY = cy - totalH/2 + lineH/2;
                            for (let ch = 0; ch < img.channels; ch++) {
                                ctx.fillText(img.typeId >= 2 ? vals[ch].toFixed(4) : vals[ch].toString(), cx, topY + ch * lineH);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Events ----
    wrap.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = wrap.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const oldScale = scale;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        scale = Math.max(0.01, Math.min(200, scale * factor));
        panX = mx - (mx - panX) * (scale / oldScale);
        panY = my - (my - panY) * (scale / oldScale);
        render();
    }, { passive: false });

    wrap.addEventListener('mousedown', (e) => {
        dragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        panStartX = panX;
        panStartY = panY;
        wrap.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (dragging) {
            panX = panStartX + (e.clientX - dragStartX);
            panY = panStartY + (e.clientY - dragStartY);
            render();
        }
        const img = curImg();
        const rect = wrap.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const col = Math.floor((mx - panX) / scale);
        const row = Math.floor((my - panY) / scale);
        if (col >= 0 && col < img.cols && row >= 0 && row < img.rows) {
            const vals = img.getPixelValues(row, col);
            const vstr = vals.map(v => typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : v).join(', ');
            pixelInfo.textContent = '[' + row + ', ' + col + '] = (' + vstr + ')';
            tooltip.style.display = 'block';
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top = (e.clientY + 12) + 'px';
            tooltip.textContent = '[' + row + ', ' + col + '] = (' + vstr + ')';
        } else {
            pixelInfo.textContent = '';
            tooltip.style.display = 'none';
        }
    });

    window.addEventListener('mouseup', () => {
        dragging = false;
        wrap.style.cursor = 'crosshair';
    });

    window.addEventListener('resize', () => { resizeCanvas(); render(); });

    // ---- Init ----
    resizeCanvas();
    buildSidebar();
    updateHeader();
    fitToView();
</script>
</body>
</html>`;
}

function deactivate() {}

module.exports = { activate, deactivate };
